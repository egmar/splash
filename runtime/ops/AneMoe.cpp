#include "ops/AneMoe.hpp"

#include "Checked.hpp"
#include "StderrLine.hpp"
#include "Supervision.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/AneMoe.h"
#include "metal/abi/QuantFormat.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

// How long an ANE evaluation may run once the GPU raised its `ready` before
// the split stops (ane::Handoff), as AneFfn's: an evaluation takes 5-60 ms,
// and Metal fails a command buffer whose event wait is still unmet
// supervision::kGpuWaitLimit after its queue reached it.
constexpr auto kHandoffBound = std::chrono::seconds(2);
static_assert(kHandoffBound < supervision::kGpuWaitLimit);
// How long the service may take to compile and load the program: as AneFfn
// allows its longest cold compile, so that a busy Mac still finishes.
constexpr auto kProgramLimit = std::chrono::seconds(210);
// How long the service may take to unload the program for the idle release
// and to load it again.
constexpr auto kReloadLimit = std::chrono::seconds(10);

// `value` up to the next multiple of `by`.
uint32_t roundUp(uint32_t value, uint32_t by) { return (value + by - 1) / by * by; }

// The MIL dimensions of `rows` rows of `width` values.
std::string dimensions(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + dimensions(rows, width) + ">";
}
// An fp16 constant of `value`, written exactly: MIL rounds it to the nearest
// fp16.
std::string fp16(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}
std::string functionName(uint32_t experts, uint32_t capacity) {
  return "moe" + std::to_string(experts) + "x" + std::to_string(capacity);
}

// The slot of each of a block's experts in its routed segments' planes, from
// its slot table (MOE_SLOT_ABSENT for an expert no slot holds).
std::vector<uint32_t> slotsOf(const BlockMoeWeights &weights) {
  if (!weights.slotTable || weights.slotTable.sizeBytes() % sizeof(uint32_t) || !weights.slotTable.contents())
    throw std::invalid_argument("ANE MoE split reads a block without a slot table");
  const auto *slots = static_cast<const uint32_t *>(weights.slotTable.contents());
  return {slots, slots + weights.slotTable.sizeBytes() / sizeof(uint32_t)};
}

// A block's scale of `expert`'s `projection` (0 gate, 1 up, 2 down), which
// its expert passes multiply the fp32 sums by (moe_expert_scale).
float expertScale(const BlockMoeWeights &weights, uint32_t experts, uint32_t expert, uint32_t projection) {
  return static_cast<const float *>(weights.expertScales.contents())[uint64_t{projection} * experts + expert];
}

// The `fraction` quantile of `values`, which it sorts descending: the count
// the residents' capacity covers at least.
uint64_t quantile(std::vector<uint64_t> &values, double fraction) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end(), std::greater<>{});
  const size_t index = std::min(values.size() - 1, static_cast<size_t>(fraction * double(values.size())));
  return values[index];
}

} // namespace

const char *AneMoe::unsupported(const MoeShape &shape) {
  if (!shape.valid()) return "ANE MoE split needs a valid MoE shape";
  if (shape.sharedSlices) return "ANE MoE split needs a routing without shared expert slices";
  if (shape.hiddenSize % 128 || shape.expertIntermediateSize % 128)
    return "ANE MoE split needs hidden and intermediate sizes of whole 128-value rows";
  return nullptr;
}

uint64_t AneMoe::expertBytes(const MoeShape &shape, uint32_t experts) {
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  uint64_t bytes = 0;
  for (const auto &[rows, width] : {std::pair{intermediate, hidden}, std::pair{intermediate, hidden},
                                    std::pair{hidden, intermediate}})
    bytes += ane::Surface::bytes(rows, width, Element::Int8) * experts;
  // A scale each group of 32 inputs a row (function()'s graph), not a row.
  bytes += ane::Surface::bytes(intermediate, hidden / 32, Element::Float16) * 2 * experts;
  bytes += ane::Surface::bytes(hidden, intermediate / 32, Element::Float16) * experts;
  return bytes;
}

uint64_t AneMoe::plannedBytes(const MoeShape &shape, const Plan &plan, uint32_t blocks) {
  if (plan.empty() || !blocks) return 0;
  const uint64_t rows = plan.rows(), hidden = shape.hiddenSize;
  // The residents' weights, then the chunk scratch: the cursors, each
  // block's mark tables, the columns' route list, the rows the pack
  // transposes and the ANE's input, scales and output, and the status word.
  uint64_t bytes = expertBytes(shape, plan.experts) * blocks;
  bytes += alignUp(uint64_t{shape.experts} * sizeof(uint32_t));
  bytes += alignUp(uint64_t{blocks} * 2 * shape.experts * sizeof(uint32_t));
  bytes += alignUp(uint64_t{blocks} * shape.experts * sizeof(uint32_t));
  bytes += alignUp(rows * sizeof(uint32_t));
  bytes += alignUp(rows * hidden * sizeof(_Float16));
  const auto width = static_cast<uint32_t>(rows);
  bytes += ane::Surface::bytes(shape.hiddenSize, width, Element::Int8);
  bytes += ane::Surface::bytes(1, width, Element::Float16);
  bytes += ane::Surface::bytes(shape.hiddenSize, width, Element::Float16);
  return bytes + alignUp(sizeof(uint32_t));
}

AneMoe::Selection AneMoe::select(std::span<const MoeShape> shapes, const Loads &loads, uint64_t budget) {
  if (shapes.empty()) throw std::invalid_argument("ANE MoE selection has no blocks");
  if (!loads.empty() && loads.size() != shapes.size())
    throw std::invalid_argument("ANE MoE selection has loads of another model");
  uint32_t most = shapes.front().experts;
  for (size_t block = 0; block < shapes.size(); ++block) {
    const MoeShape &shape = shapes[block];
    if (const char *reason = unsupported(shape)) throw std::invalid_argument(reason);
    most = std::min(most, shape.experts);
    if (!loads.empty() && !loads[block].empty() && loads[block].size() != shape.experts)
      throw std::invalid_argument("ANE MoE selection has loads of other experts");
  }
  // Each block's counts, one a piece where the corpus holds none.
  std::vector<std::vector<uint64_t>> counts(shapes.size());
  for (size_t block = 0; block < shapes.size(); ++block)
    counts[block] = loads.empty() || loads[block].empty() ? std::vector<uint64_t>(shapes[block].experts, 1)
                                                          : loads[block];
  Selection best;
  std::vector<uint64_t> selected;
  for (uint32_t experts = 1; experts <= most; experts *= 2) {
    uint64_t bytes = 0;
    for (const MoeShape &shape : shapes) bytes += expertBytes(shape, experts);
    if (bytes > budget) break;
    // The most-routed experts of each block, their counts, and the corpus's
    // routes in all.
    selected.clear();
    uint64_t total = 0;
    for (const std::vector<uint64_t> &block : counts) {
      std::vector<uint64_t> sorted = block;
      std::sort(sorted.begin(), sorted.end(), std::greater<>{});
      total += std::accumulate(block.begin(), block.end(), uint64_t{0});
      selected.insert(selected.end(), sorted.begin(), sorted.begin() + experts);
    }
    if (!total) continue;
    const uint32_t maximum = kMaximumRows / experts / kCapacityQuantum * kCapacityQuantum;
    // A capacity at a high quantile of the residents' loads holds most of
    // their routes; the most holds all of them its rows can.
    uint32_t previous = 0;
    for (const double fraction : {0.5, 0.75, 0.9, 0.95, 0.99, 1.0}) {
      const uint64_t wanted = quantile(selected, fraction);
      const uint32_t capacity =
          std::min(maximum, std::max(kMinimumRows,
                                     roundUp(static_cast<uint32_t>(std::min<uint64_t>(wanted, UINT32_MAX)),
                                             kCapacityQuantum)));
      if (!capacity || capacity == previous) continue;
      previous = capacity;
      double covered = 0.0;
      for (const uint64_t count : selected) covered += double(std::min<uint64_t>(count, capacity));
      const double fractionCovered = covered / double(total);
      if (!best.experts || fractionCovered > best.covered + 1e-9 ||
          (fractionCovered > best.covered - 1e-9 && experts * capacity < best.experts * best.capacity))
        best = {experts, capacity, bytes, fractionCovered};
    }
  }
  return best;
}

AneMoe::Plan AneMoe::plan(std::span<const MoeShape> shapes, const Loads &loads, Selection selection) {
  if (!loads.empty() && loads.size() != shapes.size())
    throw std::invalid_argument("ANE MoE plan has loads of another model");
  Plan result{.experts = selection.experts, .capacity = selection.capacity};
  if (result.empty()) return result;
  if (selection.capacity % kCapacityQuantum || uint64_t{selection.experts} * selection.capacity > kMaximumRows)
    throw std::invalid_argument("ANE MoE plan takes a capacity the blocks cannot hold");
  result.resident.resize(shapes.size());
  for (size_t block = 0; block < shapes.size(); ++block) {
    const uint32_t experts = shapes[block].experts;
    if (!experts || selection.experts > experts)
      throw std::invalid_argument("ANE MoE plan takes more experts than a block has");
    std::vector<uint32_t> ids(experts);
    std::iota(ids.begin(), ids.end(), 0u);
    if (!loads.empty() && !loads[block].empty()) {
      if (loads[block].size() != experts) throw std::invalid_argument("ANE MoE plan has loads of other experts");
      std::stable_sort(ids.begin(), ids.end(),
                       [&](uint32_t a, uint32_t b) { return loads[block][a] > loads[block][b]; });
    }
    ids.resize(selection.experts);
    result.resident[block] = std::move(ids);
  }
  return result;
}

std::string AneMoe::calibrationKey(std::span<const BlockMoeWeights *const> blocks, const MoeShape &shape) {
  if (const char *reason = unsupported(shape)) throw std::invalid_argument(reason);
  std::string key = "ane-moe calibration " + std::to_string(shape.hiddenSize) + "x" +
                    std::to_string(shape.expertIntermediateSize) + "e" + std::to_string(shape.experts);
  for (const BlockMoeWeights *block : blocks)
    key += " :g" + std::to_string(block->gate.routed.formatId) + "," + std::to_string(block->up.routed.formatId) +
           "," + std::to_string(block->down.routed.formatId);
  return key + " softcaps " + std::to_string(shape.gateSoftcap) + "/" + std::to_string(shape.upSoftcap);
}

// The ANE's expert FFN over a block's group of `experts` experts of
// `capacity` packed rows each: the group's columns are its experts' rows, a
// capacity each (x, int8 [hidden][rows], which the pack writes, and tx their
// fp16 scales), each expert's gate, up and down int8 rows at their fp16
// scale, and the output the experts' down rows, channel-major
// (AneMoeJoinParams). Every int8 weight value is dequantized by
// 1 / ANE_FFN_INT8_UNIT, which its scale carries back; the intermediate takes
// the routing's softcaps, as kernels/common/softcapped_swiglu.h computes
// them, and is divided by the headroom the down rows' scales carry.
std::string AneMoe::function(const MoeShape &shape, uint32_t experts, uint32_t capacity,
                             const std::vector<ane::Surface> &surfaces, const ane::Surface &output) {
  const uint32_t rows = experts * capacity, hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  const auto named = [](const char *prefix, uint32_t expert) { return std::string(prefix) + std::to_string(expert); };
  // x, tx and, an expert, its gate, up and down int8 rows and their scales.
  if (surfaces.size() != 2 + 6 * experts) throw std::logic_error("ANE MoE program has a group of other surfaces");
  parameters = surfaces[0].bufferType(hidden, rows) + " x, " + surfaces[1].bufferType(1, rows) + " tx";
  for (uint32_t e = 0; e < experts; ++e) {
    parameters += ", " + surfaces[2 + 6 * e].bufferType(intermediate, hidden) + " " + named("wg", e);
    parameters += ", " + surfaces[3 + 6 * e].bufferType(intermediate, hidden / 32) + " " + named("sg", e);
    parameters += ", " + surfaces[4 + 6 * e].bufferType(intermediate, hidden) + " " + named("wu", e);
    parameters += ", " + surfaces[5 + 6 * e].bufferType(intermediate, hidden / 32) + " " + named("su", e);
    parameters += ", " + surfaces[6 + 6 * e].bufferType(hidden, intermediate) + " " + named("wd", e);
    parameters += ", " + surfaces[7 + 6 * e].bufferType(hidden, intermediate / 32) + " " + named("sd", e);
  }
  line(tensor("int8", hidden, rows) + " x_t = tensor_buffer_to_tensor<ios17>(input = x)");
  line(tensor("fp16", 1, rows) + " tx_t = tensor_buffer_to_tensor<ios17>(input = tx)");
  f16("x_u", hidden, rows, "dequantize(input = x_t, scale = " + fp16(1.0) + ")");
  f16("x_d", hidden, rows, "mul(x = x_u, y = tx_t)");
  std::string outputs;
  for (uint32_t e = 0; e < experts; ++e) {
    const std::string g = named("wg", e), u = named("wu", e), d = named("wd", e);
    const std::string s = std::to_string(e), sg = named("sg", e), su = named("su", e), sd = named("sd", e);
    line(tensor("int8", intermediate, hidden) + " " + g + "_t = tensor_buffer_to_tensor<ios17>(input = " + g + ")");
    line(tensor("int8", intermediate, hidden) + " " + u + "_t = tensor_buffer_to_tensor<ios17>(input = " + u + ")");
    line(tensor("int8", hidden, intermediate) + " " + d + "_t = tensor_buffer_to_tensor<ios17>(input = " + d + ")");
    f16(g + "_d", intermediate, hidden, "dequantize(input = " + g + "_t, scale = " + unit + ")");
    f16(u + "_d", intermediate, hidden, "dequantize(input = " + u + "_t, scale = " + unit + ")");
    f16(d + "_d", hidden, intermediate, "dequantize(input = " + d + "_t, scale = " + unit + ")");
    line(tensor("fp16", intermediate, hidden / 32) + " " + sg + "_t = tensor_buffer_to_tensor<ios17>(input = " + sg + ")");
    line(tensor("fp16", intermediate, hidden / 32) + " " + su + "_t = tensor_buffer_to_tensor<ios17>(input = " + su + ")");
    line(tensor("fp16", hidden, intermediate / 32) + " " + sd + "_t = tensor_buffer_to_tensor<ios17>(input = " + sd + ")");
    // Each group of 32 inputs' codes come back at its own scale before the
    // matmul: a group's outliers cost their group's precision, not its row's
    // (AneMoe.cpp's graph; a scale a row's would be one multiply after the
    // matmul, and a quarter more quantisation error).
    const auto grouped = [&](const std::string &codes, const std::string &scale, const std::string &scaled,
                             uint32_t height, uint32_t width) {
      const std::string h = std::to_string(height), w = std::to_string(width), g = std::to_string(width / 32);
      line("tensor<fp16, [1, 1, " + h + ", " + g + ", 1]> " + scale + "_5 = reshape(x = " + scale +
           "_t, shape = tensor<int32, [5]>([1, 1, " + h + ", " + g + ", 1]))");
      line("tensor<fp16, [1, 1, " + h + ", " + g + ", 32]> " + codes + "_5 = reshape(x = " + codes +
           "_d, shape = tensor<int32, [5]>([1, 1, " + h + ", " + g + ", 32]))");
      line("tensor<fp16, [1, 1, " + h + ", " + g + ", 32]> " + scaled + "_5 = mul(x = " + codes + "_5, y = " +
           scale + "_5)");
      f16(scaled, height, width,
          "reshape(x = " + scaled + "_5, shape = tensor<int32, [4]>(" + dimensions(height, width) + "))");
    };
    grouped(g, sg, g + "_w", intermediate, hidden);
    grouped(u, su, u + "_w", intermediate, hidden);
    grouped(d, sd, d + "_w", hidden, intermediate);
    const std::string x = "x" + s;
    if (experts > 1)
      line(tensor("fp16", hidden, capacity) + " " + x +
           " = slice_by_size(x = x_d, begin = tensor<int32, [4]>([0, 0, 0, " + std::to_string(e * capacity) +
           "]), size = tensor<int32, [4]>(" + dimensions(hidden, capacity) + "))");
    const std::string in = experts > 1 ? x : "x_d";
    f16(g + "m", intermediate, capacity,
        "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + g + "_w, y = " + in + ")");
    f16(u + "m", intermediate, capacity,
        "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + u + "_w, y = " + in + ")");
    // tanh(g / cap) * cap * sigmoid(g): the gate's softcap, or g where the
    // routing has none; sigmoid(g) as (1 + tanh(g / 2)) / 2 through the ANE's
    // tanh, as AneFfn's SiLU.
    if (shape.gateSoftcap) {
      f16(g + "c", intermediate, capacity, "mul(x = " + g + "m, y = " + fp16(1.0 / shape.gateSoftcap) + ")");
      f16(g + "ct", intermediate, capacity, "tanh(x = " + g + "c)");
      f16(g + "cap", intermediate, capacity, "mul(x = " + g + "ct, y = " + fp16(shape.gateSoftcap) + ")");
    }
    f16(g + "h", intermediate, capacity, "mul(x = " + g + "m, y = " + fp16(0.5) + ")");
    f16(g + "t", intermediate, capacity, "tanh(x = " + g + "h)");
    f16(g + "p", intermediate, capacity, "add(x = " + g + "t, y = " + fp16(1.0) + ")");
    f16(g + "g", intermediate, capacity, "mul(x = " + g + "p, y = " + fp16(0.5) + ")");
    f16(g + "f", intermediate, capacity,
        "mul(x = " + (shape.gateSoftcap ? g + "cap" : g + "m") + ", y = " + g + "g)");
    if (shape.upSoftcap) {
      f16(u + "c", intermediate, capacity, "mul(x = " + u + "m, y = " + fp16(1.0 / shape.upSoftcap) + ")");
      f16(u + "ct", intermediate, capacity, "tanh(x = " + u + "c)");
      f16(u + "cap", intermediate, capacity, "mul(x = " + u + "ct, y = " + fp16(shape.upSoftcap) + ")");
    }
    f16("h" + s, intermediate, capacity,
        "mul(x = " + g + "f, y = " + (shape.upSoftcap ? u + "cap" : u + "m") + ")");
    f16("hh" + s, intermediate, capacity, "mul(x = h" + s + ", y = " + fp16(1.0 / kHeadroom) + ")");
    f16("y" + s, hidden, capacity,
        "matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + d + "_w, y = hh" + s + ")");
    outputs += std::string(outputs.empty() ? "" : ", ") + "y" + s;
  }
  if (experts > 1)
    f16("yall", hidden, rows,
        "concat(axis = int32(3), interleave = bool(false), values = (" + outputs + "))");
  line(output.bufferType(hidden, rows) + " y = tensor_to_tensor_buffer<ios17>(input = " +
       (experts > 1 ? "yall" : outputs) +
       ", interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" +
       output.strides(hidden) + "))");
  return "    func " + functionName(experts, capacity) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

AneMoe::AneMoe(metal::MetalBackend &backend, std::span<const BlockMoeWeights *const> blocks, const MoeShape &shape,
               Plan plan, Loads loads, std::function<bool()> interrupted)
    : backend_(backend), blocks_(blocks.begin(), blocks.end()), shape_(shape), plan_(std::move(plan)),
      handoff_(backend, kHandoffBound) {
  if (const char *reason = unsupported(shape)) throw std::invalid_argument(reason);
  if (!plan_.resident.empty() && plan_.resident.size() != blocks.size())
    throw std::invalid_argument("ANE MoE split has residents of other blocks");
  if (!plan_.empty() &&
      (!plan_.experts || plan_.capacity % kCapacityQuantum || plan_.rows() > kMaximumRows ||
       plan_.experts > shape.experts))
    throw std::invalid_argument("ANE MoE split takes experts or capacity the blocks cannot hold");
  for (const std::vector<uint32_t> &resident : plan_.resident) {
    if (resident.empty()) continue;
    if (resident.size() != plan_.experts)
      throw std::invalid_argument("ANE MoE split's blocks hold other counts of experts");
    for (const uint32_t expert : resident)
      if (expert >= shape.experts) throw std::invalid_argument("ANE MoE split has an expert a block lacks");
  }
  for (const BlockMoeWeights *block : blocks_) {
    if (!block) throw std::invalid_argument("ANE MoE split has no block");
    if (block->stream) throw std::invalid_argument("ANE MoE split needs resident expert blocks");
    for (const BlockExpertProjection *projection : {&block->gate, &block->up, &block->down}) {
      if (projection->routed.isFloat() || projection->routed.formatId >= GGUF_FMT_COUNT)
        throw std::invalid_argument("ANE MoE split needs quantized expert projections");
      requireSegmentPlanes(projection->routed, "ANE MoE expert projection");
    }
    requireBytes(block->expertScales, uint64_t{3} * shape.experts * sizeof(float), "ANE MoE expert scales");
    if (!block->expertScales.contents())
      throw std::invalid_argument("ANE MoE expert scales are not visible to the CPU");
  }
  const auto buffer = [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer made = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    allocatedBytes_ += made.allocatedBytes();
    return made;
  };
  const auto surface = [&](uint32_t rows, uint32_t width, Element element) {
    ane::Surface created = ane::Surface::create(backend_, rows, width, element);
    allocatedBytes_ += created.buffer.allocatedBytes();
    return created;
  };
  // A split of no block takes no command: nothing to allocate, compile or
  // stage, and splits() is false.
  if (plan_.empty()) return;
  const uint32_t hidden = shape_.hiddenSize, intermediate = shape_.expertIntermediateSize, rows = plan_.rows();
  // The chunk scratch every application writes, in the rows the plan's group
  // holds: the columns' route list and scales, the rows the pack transposes,
  // and the ANE's input and output surfaces.
  cursors_ = buffer(uint64_t{shape_.experts} * sizeof(uint32_t), "ane moe cursors");
  routeList_ = buffer(uint64_t{rows} * sizeof(uint32_t), "ane moe route list");
  std::memset(routeList_.contents(), 0xFF, uint64_t{rows} * sizeof(uint32_t));
  scaled_ = buffer(uint64_t{rows} * hidden * sizeof(_Float16), "ane moe scaled rows");
  status_ = buffer(sizeof(uint32_t), "ane moe status");
  *static_cast<uint32_t *>(status_.contents()) = 0;
  // The routes each block's experts take, which finish() reads: the counts a
  // next start plans its residents from.
  counts_ = buffer(uint64_t{blocks.size()} * shape_.experts * sizeof(uint32_t), "ane moe expert counts");
  std::memset(counts_.contents(), 0, counts_.sizeBytes());
  if (loads.empty())
    loads_.assign(blocks.size(), std::vector<uint64_t>(shape_.experts, 0));
  else if (loads.size() == blocks.size() &&
           std::ranges::all_of(loads, [&](const std::vector<uint64_t> &block) { return block.size() == shape_.experts; }))
    loads_ = std::move(loads);
  else
    throw std::invalid_argument("ANE MoE split's loads are of other blocks or experts");
  x_ = surface(hidden, rows, Element::Int8);
  tx_ = surface(1, rows, Element::Float16);
  y_ = surface(hidden, rows, Element::Float16);

  groups_.resize(blocks.size());
  // The surfaces of the first block's group, which the program's parameters
  // are: every block's group holds the plan's rows, so the strides agree.
  std::vector<ane::Surface> groupSurfaces = {x_, tx_};
  for (size_t block = 0; block < blocks.size(); ++block) {
    Group &group = groups_[block];
    group.experts = plan_.resident[block];
    if (group.experts.empty()) continue;
    std::vector<uint32_t> base(shape_.experts, ANE_MOE_EXPERT_ABSENT), capacity(shape_.experts, 0);
    for (size_t slot = 0; slot < group.experts.size(); ++slot) {
      // A scale a group of 32 inputs a row (AneMoe.cpp's graph multiplies
      // each group's codes before the matmul), which costs a quarter of the
      // per-row scales' quantisation error on an expert's rows.
      group.gate.push_back(surface(intermediate, hidden, Element::Int8));
      group.gateScale.push_back(surface(intermediate, hidden / 32, Element::Float16));
      group.up.push_back(surface(intermediate, hidden, Element::Int8));
      group.upScale.push_back(surface(intermediate, hidden / 32, Element::Float16));
      group.down.push_back(surface(hidden, intermediate, Element::Int8));
      group.downScale.push_back(surface(hidden, intermediate / 32, Element::Float16));
      base[group.experts[slot]] = static_cast<uint32_t>(slot) * plan_.capacity;
      capacity[group.experts[slot]] = plan_.capacity;
    }
    group.baseTable = buffer(uint64_t{shape_.experts} * sizeof(uint32_t), "ane moe expert base");
    group.capacityTable = buffer(uint64_t{shape_.experts} * sizeof(uint32_t), "ane moe expert capacity");
    std::memcpy(group.baseTable.contents(), base.data(), base.size() * sizeof(uint32_t));
    std::memcpy(group.capacityTable.contents(), capacity.data(), capacity.size() * sizeof(uint32_t));
    if (groupSurfaces.size() == 2)
      for (size_t slot = 0; slot < group.experts.size(); ++slot)
        groupSurfaces.insert(groupSurfaces.end(),
                             {group.gate[slot], group.gateScale[slot], group.up[slot], group.upScale[slot],
                              group.down[slot], group.downScale[slot]});
  }
  // One function serves every block: the plan holds one group shape.
  program_ = std::make_unique<ane::Program>(
      "program(1.3)\n{\n" + function(shape_, plan_.experts, plan_.capacity, groupSurfaces, y_) + "}\n",
      std::span<const uint8_t>{}, ane::Program::Limits{kProgramLimit, std::move(interrupted)});
  const uint32_t procedure = program_->procedure(functionName(plan_.experts, plan_.capacity));
  for (size_t block = 0; block < blocks.size(); ++block) {
    Group &group = groups_[block];
    if (group.experts.empty()) continue;
    // Each block's group's surfaces by the program's names, in the order the
    // program's own copy of them takes them (Program::inputs), which need not
    // be the MIL's.
    std::vector<std::pair<std::string, const ane::Surface *>> named = {{"x", &x_}, {"tx", &tx_}};
    for (uint32_t e = 0; e < plan_.experts; ++e) {
      const std::string s = std::to_string(e);
      named.insert(named.end(), {{"wg" + s, &group.gate[e]}, {"sg" + s, &group.gateScale[e]},
                                 {"wu" + s, &group.up[e]}, {"su" + s, &group.upScale[e]},
                                 {"wd" + s, &group.down[e]}, {"sd" + s, &group.downScale[e]}});
    }
    std::vector<ane::Surface> bound;
    for (const std::string &name : program_->inputs(procedure)) {
      const auto found = std::find_if(named.begin(), named.end(),
                                      [&](const auto &entry) { return entry.first == name; });
      if (found == named.end()) throw std::logic_error("ANE MoE program has an unknown input " + name);
      bound.push_back(*found->second);
    }
    group.binding = program_->bind(procedure, bound, y_);
  }
  // Every resident expert's int8 rows and scales, on the GPU, from its block's
  // planes: gate's and up's rows over the hidden inputs, down's over the
  // intermediate ones.
  metal::CommandGraph graph;
  for (size_t block = 0; block < blocks.size(); ++block) {
    const Group &group = groups_[block];
    if (group.experts.empty()) continue;
    const BlockMoeWeights &weights = *blocks[block];
    const std::vector<uint32_t> slots = slotsOf(weights);
    for (size_t at = 0; at < group.experts.size(); ++at) {
      const uint32_t expert = group.experts[at], slot = slots.at(expert);
      if (slot == MOE_SLOT_ABSENT)
        throw std::invalid_argument("ANE MoE split's resident expert holds no slot");
      const auto add = [&](const BlockExpertProjection &projection, const ane::Surface &output,
                           const ane::Surface &scale, uint32_t row, uint32_t width, uint32_t groups,
                           uint32_t height, float multiplier) {
        graph.add("ane_moe_weights_grouped_gguf",
                  {projection.routed.plane0, projection.routed.plane1Slot(), projection.routed.meta, output.buffer,
                   scale.buffer},
                  AneMoeWeightParams{groups, row, 0, width, output.strideBytes,
                                     static_cast<uint32_t>(scale.strideBytes / sizeof(_Float16)),
                                     projection.routed.formatId, height, multiplier},
                  {(height + ANE_FFN_WEIGHT_ROWS - 1) / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
      };
      add(weights.gate, group.gate[at], group.gateScale[at], slot * intermediate, hidden, hidden / 32, intermediate,
          expertScale(weights, shape_.experts, expert, 0));
      add(weights.up, group.up[at], group.upScale[at], slot * intermediate, hidden, hidden / 32, intermediate,
          expertScale(weights, shape_.experts, expert, 1));
      add(weights.down, group.down[at], group.downScale[at], slot * hidden, intermediate, intermediate / 32, hidden,
          expertScale(weights, shape_.experts, expert, 2) * float(kHeadroom));
    }
  }
  if (!graph.dispatches().empty()) static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
}

uint32_t AneMoe::blockOf(const BlockMoeWeights &weights) const {
  const auto found = std::find(blocks_.begin(), blocks_.end(), &weights);
  if (found == blocks_.end())
    throw std::logic_error("ANE MoE split was asked to split a block it was not built from");
  return static_cast<uint32_t>(found - blocks_.begin());
}

bool AneMoe::splits(uint32_t rows) const {
  return !plan_.empty() && rows >= kMinimumRows && rows <= kMaximumRows && !handoff_.retired();
}

bool AneMoe::begin(uint32_t rows) {
  rows_ = rows;
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  pending_.reset();
  return splits(rows);
}

void AneMoe::addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers, const MoePlan &plan,
                        const BlockMoeWeights &weights) {
  if (!splits(plan.rows())) return;
  const uint32_t block = blockOf(weights);
  if (groups_[block].experts.empty()) return;
  const Group &group = groups_[block];
  if (plan.rows() != rows_)
    throw std::logic_error("ANE MoE split packs a command of other rows than it began");
  if (pending_) throw std::logic_error("ANE MoE split adds an application before it joined the one before");
  const MoeShape &shape = shape_;
  const uint64_t routes = uint64_t{rows_} * shape.routesPerToken();
  requireBytes(buffers.scratch.selectedExperts, routes * sizeof(uint32_t), "ANE MoE selected experts");
  requireBytes(buffers.scratch.routeRows, routes * sizeof(uint32_t), "ANE MoE route rows");
  requireBytes(buffers.input, uint64_t{rows_} * shape.hiddenSize * sizeof(uint16_t), "ANE MoE input");
  requireBytes(buffers.scratch.expertOutput,
               (routes + plan_.rows()) * shape.hiddenSize * sizeof(uint16_t), "ANE MoE expert output");
  // The block's cursors, the group's scales and route list, then its routes
  // marked out of the selection: the grouping skips those the ANE takes.
  graph.add("ane_moe_clear", {cursors_, tx_.buffer, routeList_},
            AneMoeClearParams{shape.experts, plan_.rows(), 0}, {1, 1, 1});
  graph.add("ane_moe_mark",
            {buffers.scratch.selectedExperts, buffers.scratch.routeRows, group.baseTable, group.capacityTable,
             cursors_, routeList_, counts_},
            AneMoeMarkParams{static_cast<uint32_t>(routes), shape.experts, static_cast<uint32_t>(routes), block},
            {(static_cast<uint32_t>(routes) + 255) / 256, 1, 1});
  graph.add("ane_moe_scale", {buffers.input, routeList_, scaled_, tx_.buffer},
            AneMoeScaleParams{shape.routesPerToken(), shape.hiddenSize, 0, plan_.rows()}, {plan_.rows(), 1, 1});
  graph.add("ane_ffn_pack", {scaled_, x_.buffer}, AneFfnPackParams{shape.hiddenSize, 0, x_.strideBytes},
            {plan_.rows() / ANE_FFN_TILE, shape.hiddenSize / ANE_FFN_TILE, 1},
            {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  const uint64_t ready = handoff_.next(), done = handoff_.next();
  graph.signal(handoff_.event(), ready);
  jobs_.push_back({block, ready, done});
  pending_ = block;
}

void AneMoe::addJoin(metal::CommandGraph &graph, const MoeBuffers &buffers, const MoePlan &plan,
                     const BlockMoeWeights &weights) {
  if (!splits(plan.rows()) || !pending_) return;
  const uint32_t block = blockOf(weights);
  if (*pending_ != block)
    throw std::logic_error("ANE MoE split joins block " + std::to_string(block) + " after " +
                           std::to_string(*pending_));
  if (plan.rows() != rows_)
    throw std::logic_error("ANE MoE split joins a command of other rows than it began");
  pending_.reset();
  graph.wait(handoff_.event(), jobs_.back().done);
  graph.add("ane_moe_join", {y_.buffer, buffers.scratch.expertOutput, status_},
            AneMoeJoinParams{static_cast<uint32_t>(uint64_t{rows_} * shape_.routesPerToken()), 0, plan_.rows(),
                             shape_.hiddenSize, y_.strideBytes / uint32_t{sizeof(_Float16)}},
            {plan_.rows(), 1, 1});
}

metal::CommandTicket AneMoe::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  if (pending_) throw std::logic_error("ANE MoE split commits an application it did not join");
  if (jobs_.size() > kMostInFlight)
    throw std::logic_error("ANE MoE command queues " + std::to_string(jobs_.size()) +
                           " evaluations, more than the Neural Engine holds in flight");
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  unfinished_ = !jobs.empty();
  committedApplications_ = static_cast<uint32_t>(jobs.size());
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    handoff_.cancel(reason);
    warnStopped();
  };
  metal::CommandTicket ticket;
  try {
    ticket = backend_.submitCommandAsync(graph.command(), std::move(completion));
  } catch (const std::exception &error) {
    cancel(std::string("its Metal command was not submitted: ") + error.what());
    throw;
  } catch (...) {
    cancel("its Metal command was not submitted");
    throw;
  }
  for (const Job &job : jobs) queue(job);
  return ticket;
}

bool AneMoe::finish(const metal::CommandTiming &) {
  unfinished_ = false;
  std::optional<ane::Handoff::Ran> ran = handoff_.finish();
  auto *status = static_cast<uint32_t *>(status_.contents());
  if (status && std::exchange(*status, 0u) && ran) {
    handoff_.retire("the Neural Engine's output was not finite");
    ran.reset();
  }
  if (!ran) {
    warnStopped();
    return false;
  }
  // The routes this command's marks counted, added to the split's, and the
  // buffer cleared for the next command's command to write.
  if (const auto *counts = static_cast<const uint32_t *>(counts_.contents())) {
    for (size_t block = 0; block < loads_.size(); ++block)
      for (uint32_t expert = 0; expert < loads_[block].size(); ++expert)
        loads_[block][expert] += counts[block * loads_[block].size() + expert];
    std::memset(counts_.contents(), 0, counts_.sizeBytes());
  }
  ++served_.commands;
  served_.evaluations += committedApplications_;
  return true;
}

std::string AneMoe::encodeLoads(const Loads &loads) {
  std::string line = "ane-moe loads";
  for (const std::vector<uint64_t> &block : loads)
    for (const uint64_t count : block) line += " " + std::to_string(count);
  return line;
}

std::optional<AneMoe::Loads> AneMoe::parseLoads(std::string_view line, uint32_t blocks, uint32_t experts) {
  constexpr std::string_view prefix = "ane-moe loads";
  if (line.substr(0, prefix.size()) != prefix) return std::nullopt;
  line.remove_prefix(prefix.size());
  Loads loads(blocks, std::vector<uint64_t>(experts, 0));
  size_t read = 0;
  while (!line.empty()) {
    const size_t start = line.find_first_not_of(' ');
    if (start == std::string_view::npos) break;
    line.remove_prefix(start);
    if (read == uint64_t{blocks} * experts) return std::nullopt;
    uint64_t count = 0;
    const auto [end, error] = std::from_chars(line.data(), line.data() + line.size(), count);
    if (error != std::errc{}) return std::nullopt;
    loads[read / experts][read % experts] = count;
    ++read;
    line.remove_prefix(static_cast<size_t>(end - line.data()));
  }
  if (read != uint64_t{blocks} * experts) return std::nullopt;
  return loads;
}

bool AneMoe::release() {
  if (!program_) return false;
  if (unfinished_ || !jobs_.empty())
    throw std::logic_error("ANE MoE split released while a command it encoded is unfinished");
  if (released_ || (handoff_.retired() && !handoff_.idle())) return false;
  try {
    program_->unload({kReloadLimit, {}});
  } catch (const std::exception &error) {
    handoff_.retire(std::string("its program did not unload: ") + error.what());
    warnStopped();
    return false;
  }
  released_ = true;
  logLine("Neural Engine MoE program unloaded while idle");
  return true;
}

void AneMoe::restore() noexcept {
  if (!program_ || !released_ || handoff_.retired()) return;
  released_ = false;
  std::string failure = "an unknown exception";
  try {
    program_->load({kReloadLimit, {}});
    return;
  } catch (const std::exception &error) {
    failure = error.what();
  } catch (...) {
  }
  handoff_.retire("its program did not load again: " + failure);
  warnStopped();
}

void AneMoe::warnStopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("Neural Engine MoE split stopped (", handoff_.reason(),
             "); the GPU runs the prefill MoE alone until the engine restarts.");
}

void AneMoe::queue(const Job &job) {
  const ane::Program::Binding &binding = *groups_.at(job.block).binding;
  handoff_.queue(job.ready, job.done,
                 [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                   program_->enqueue(binding, event, wait, signal, std::move(report));
                 });
}

} // namespace splash::ops
