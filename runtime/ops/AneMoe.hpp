#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/MoE.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::ops {

// The routed experts of the model's MoE blocks that the Neural Engine holds,
// and the split of a prefill chunk's routes with it: the routes of an expert
// the engine holds are packed into the ANE's int8 input rows and leave the
// GPU's grouping, which skips them (metal/abi/AneMoe.h); the ANE's expert
// outputs are joined into the rows the combine reads, in the one prefill
// command, before the combine. The GPU runs the remaining routes, and the
// whole block where the engine holds none of its experts.
//
// One program serves every block: its evaluations then report in order, as
// AneFfn's do, so the handoff's one shared event orders each join against its
// own evaluation (a program a block would let an early block's signal meet
// another's wait — the fault the benchmark's pipeline mode found). Every
// block's group is `plan.experts` experts of `plan.capacity` packed rows each
// and has the same rows, the program's function: a chunk of 48 applications
// queues one evaluation each, within the 127 the engine holds in flight.
//
// The weights are int8 copies the GPU requantizes once at construction from
// the blocks' GGUF planes, without the dense split's Hadamard rotation: an
// expert's weight rows spread less than a dense FFN's, and the rotation's
// Metal kernel a row costs more than it earns here. A row's scale carries the
// expert's scale and, for down, the headroom that keeps fp16's product in
// range (AneMoe.cpp). Any failure of the ANE's work stops the split for the
// life of the process; finish() then reports the command's results unusable
// and the caller runs the command again on the GPU alone, which every later
// command runs too.
class AneMoe final {
public:
  // A group's capacity rounds up to it: the pack writes whole 32-row tiles.
  static constexpr uint32_t kCapacityQuantum = 32;
  // The least and most rows a block's group holds, its program function's
  // rows: its experts' capacities, not a chunk's rows. 8,192 is eight experts
  // of 1,024 packed rows, whose surfaces take 113 MiB at Inkling-2-Mini's
  // hidden size and whose evaluation runs about as long as the GPU's whole
  // block, past which the join would wait on the Neural Engine rather than
  // the other way round (the benchmark's graphs were timed to 4,096 rows).
  static constexpr uint32_t kMinimumRows = kCapacityQuantum, kMaximumRows = 8192;
  // The most evaluations the ANE's client holds in flight for a program, and
  // so the most applications a chunk splits.
  static constexpr uint32_t kMostInFlight = 127;
  // The intermediate values' headroom through the down projection, as the
  // benchmark's graph takes it: the down rows' scales carry it, and the
  // graph divides the intermediate by it, so that fp16's matmul sum of
  // softcapped products stays in range.
  static constexpr uint32_t kHeadroom = 8;

  // Which experts of each block the engine holds. A block's residents are
  // `experts` of its experts of `capacity` packed rows each; `resident` holds
  // them per block, in the blocks' order, none for a block the split leaves
  // on the GPU.
  struct Plan final {
    uint32_t experts = 0, capacity = 0;
    std::vector<std::vector<uint32_t>> resident;
    [[nodiscard]] uint32_t rows() const noexcept { return experts * capacity; }
    [[nodiscard]] bool empty() const noexcept { return !experts || !capacity; }
  };

  // Each block's routed-expert counts over a calibration corpus: counts[block][expert].
  using Loads = std::vector<std::vector<uint64_t>>;

  // The best plan a budget of resident int8 weights buys over `loads`: a
  // grid of expert counts and capacities, by the routes it holds of the
  // corpus. None when the budget holds no block's smallest group.
  struct Selection final {
    uint32_t experts = 0, capacity = 0;
    uint64_t bytes = 0;
    double covered = 0.0; // the fraction of the corpus's routes it holds
    [[nodiscard]] bool empty() const noexcept { return !experts || !capacity; }
  };
  // `loads` may be empty (no corpus yet): every expert then counts alike, and
  // any experts serve. Throws std::invalid_argument for loads of other blocks
  // or experts than the shapes'.
  [[nodiscard]] static Selection select(std::span<const MoeShape> shapes, const Loads &loads, uint64_t budget);
  // The plan of `selection` over `loads`: each block's most-routed experts,
  // by load, `selection.experts` of them.
  [[nodiscard]] static Plan plan(std::span<const MoeShape> shapes, const Loads &loads, Selection selection);
  // The int8 weights of `experts` experts of a block, and their scales: what
  // the split allocates for them.
  [[nodiscard]] static uint64_t expertBytes(const MoeShape &shape, uint32_t experts);

  // Why the split does not take blocks of `shape`, or null when it does: a
  // resident block of a routing without shared slices, unrotated quantized
  // expert projections of whole 128-input rows (the weight kernel writes
  // four, a simdgroup).
  [[nodiscard]] static const char *unsupported(const MoeShape &shape);
  // What tells this Mac's calibrations apart for one macOS version, engine
  // build and budget: the shapes and formats of the blocks' resident experts.
  [[nodiscard]] static std::string calibrationKey(std::span<const BlockMoeWeights> blocks, const MoeShape &shape);

  // The split of `blocks` (the model's MoE blocks, resident in memory, in the
  // weight layers' order) with the experts `plan` names: their int8 weights
  // are requantized here, on the GPU, and the program compiled and loaded.
  // The wait for the ANE's service ends with ane::Interrupted once
  // `interrupted` returns true. Throws std::invalid_argument for a plan the
  // blocks or shapes cannot hold, or an unsupported shape.
  AneMoe(metal::MetalBackend &backend, std::span<const BlockMoeWeights> blocks, const MoeShape &shape, Plan plan,
         std::function<bool()> interrupted);
  AneMoe(const AneMoe &) = delete;
  AneMoe &operator=(const AneMoe &) = delete;

  [[nodiscard]] const Plan &plan() const noexcept { return plan_; }
  [[nodiscard]] const MoeShape &shape() const noexcept { return shape_; }
  // The commands finish() found split since the split was built, and their
  // evaluations.
  struct Served final {
    uint64_t commands = 0, evaluations = 0;
  };
  [[nodiscard]] const Served &served() const noexcept { return served_; }
  // The rows of a block's group, its applications' rows of the scratch's
  // expert outputs: 0 for a block the split leaves on the GPU.
  [[nodiscard]] uint32_t regionRows(uint32_t block) const noexcept;
  // What Metal allocated for the split.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // Whether the split takes a command of `rows` rows: a prefill chunk from
  // kMinimumRows up, its evaluations within kMostInFlight, and not once it
  // stopped.
  [[nodiscard]] bool splits(uint32_t rows) const;
  // Whether the split stopped, and why.
  [[nodiscard]] bool retired() const { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const { return handoff_.reason(); }

  // Starts a command of `rows` rows; false once the split stopped. A command
  // begun but neither committed nor cancelled is cancelled by the next one.
  bool begin(uint32_t rows);
  // After block `block`'s routing, before its GPU work: the ANE's routes
  // marked out of `buffers.scratch.selectedExperts`, their input rows packed
  // into the ANE's surface, and the block's evaluation signalled — one
  // `addRouting` an application, its `addJoin` after the block's expert
  // passes and before its combine. Nothing for a block the split leaves on
  // the GPU, or a command the split does not take.
  void addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers, const MoePlan &plan, uint32_t block);
  // Waits for the application's evaluation and copies its rows into
  // `buffers.scratch.expertOutput`, which must hold the block's region after
  // the routes (regionRows()).
  void addJoin(metal::CommandGraph &graph, const MoeBuffers &buffers, const MoePlan &plan, uint32_t block);
  // Submits the command encoded since begin() (MetalBackend::submitCommandAsync),
  // then starts its evaluations. A submission that throws starts none and
  // stops the split before the exception goes on. Throws std::logic_error for
  // an application not joined.
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph, metal::CommandCompletion completion);
  // After the command committed last has completed: whether its outputs hold
  // the split's expert outputs. False when the ANE's work failed or wrote a
  // value that is not finite, which stops the split and logs why, once.
  [[nodiscard]] bool finish(const metal::CommandTiming &timing = {});

  // The engine's idle release (engine::ReleasableMemory): unloads the
  // program, whose memory the ANE service gives back, and keeps the split's
  // own: the resident int8 weights, which a reload does not compile again.
  // True once it unloaded the program; false, doing nothing, when it is
  // unloaded, or the split stopped while evaluations it gave up on may still
  // use it. Throws std::logic_error while a command it encoded is unfinished.
  [[nodiscard]] bool release();
  // Loads the program release() unloaded again, unless the split stopped.
  // Never throws: a program that does not load stops the split, and the GPU
  // runs every later chunk alone.
  void restore() noexcept;

private:
  // A block's group: its residents' ids, in the packed rows' order, the
  // surfaces of their int8 weights and scales, and the tables the mark kernel
  // reads (a uint32 an expert of the block).
  struct Group final {
    std::vector<uint32_t> experts;
    std::vector<ane::Surface> gate, up, down, gateScale, upScale, downScale;
    metal::MetalBuffer baseTable, capacityTable;
    std::optional<ane::Program::Binding> binding;
  };
  // An application's evaluation of `block`: it starts once the event reaches
  // `ready` and raises it to `done`.
  struct Job final {
    uint32_t block;
    uint64_t ready, done;
  };

  // The program's function over a group of `experts` experts of `capacity`
  // packed rows.
  [[nodiscard]] static std::string function(const MoeShape &shape, uint32_t experts, uint32_t capacity,
                                            const std::vector<ane::Surface> &surfaces, const ane::Surface &output);
  // Requantizes `block`'s residents into `group`'s surfaces, from the block's
  // GGUF planes.
  void stage(metal::CommandGraph &graph, uint32_t block, const Group &group) const;
  // Starts `job` on the ANE through the handoff.
  void queue(const Job &job);
  // Logs, once, that the split stopped.
  void warnStopped();

  metal::MetalBackend &backend_;
  MoeShape shape_;
  Plan plan_;
  std::vector<Group> groups_;
  // The chunk scratch every application writes: the marks' cursors, the
  // packed columns' route list and scales, the rows the pack transposes, the
  // ANE's input and output surfaces, and the status word the join sets.
  metal::MetalBuffer cursors_, routeList_, scaled_, status_;
  ane::Surface x_, tx_, y_;
  std::unique_ptr<ane::Program> program_;
  // The rows of the command being encoded, and its applications: `jobs_`
  // commit() starts, `pending_` is the application added but not joined.
  uint32_t rows_ = 0;
  std::vector<Job> jobs_;
  std::optional<uint32_t> pending_;
  uint32_t committedApplications_ = 0;
  // Whether a command committed jobs that finish() has not yet judged.
  bool unfinished_ = false;
  bool released_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
  Served served_;
  // Destroyed first: it waits for the evaluations the ANE has not reported.
  ane::Handoff handoff_;
};

} // namespace splash::ops
