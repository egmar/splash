// The routed experts' Neural Engine split of a prefill MoE (ops/AneMoe.cpp,
// kernels/shared/ane_moe.metal):
// - the plan: the experts and capacity a budget of resident int8 weights buys
//   over a corpus's counts (ops::AneMoe::select), the most-routed experts it
//   names (plan), and the key a Mac's calibration is kept under;
// - the split, on a block of four experts in GGUF formats: one command of
//   routes marks the residents' routes out of the selection up to each
//   expert's capacity, packs their input rows, runs the ANE's group, and joins
//   its outputs into the rows the combine reads, as the block's experts'
//   forward of those rows computes them; the routes past capacity and the
//   other experts' stay the GPU's, their route rows untouched, and a padding
//   column's rows are zero; a second command computes what the first did, and
//   a split whose plan takes no block takes no command;
// - the step (ops::MoE::addRouting and addRouted with the split): a whole MoE
//   step of a block of eight experts — router, selection, grouping, expert
//   passes and combine — computes what it computes without the split within
//   the Neural Engine's error, its routes are the engine's and the GPU's, and
//   a step's workspace holds the engine's region (MoeConfig::aneRows).
// The kernels round where the CPU may not: an expert's int8 rows cost about
// 1% and the ANE's fp16 arithmetic the rest of the ~5% a joined row differs by
// (kMaximumError), which a step's combine averages to 0.6%.
#include "LinearNumerics.hpp"
#include "MoeFixture.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/AneMoe.h"
#include "ops/AneMoe.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

namespace {

constexpr uint32_t kHidden = 256, kIntermediate = 128, kExperts = 4, kTopK = 2;
constexpr uint32_t kRows = 64, kCapacity = 32, kResidentExperts = 2, kStepExperts = 8;
// The routes of the block's chunk: 20 to each resident expert (expert 1 first,
// then expert 3), which the capacity holds, and the rest to the others.
constexpr uint32_t kRoutesToResident = 20;
// The ANE's group differs from the FP32 forward of the same weights by about
// 4% RMS a row with these weights: int8 rows (which the dense split measures
// at 2.5-2.8% against the GPU's) and the ANE's fp16 arithmetic, which the
// benchmark's check reports diluted over its padding (0.65% over a run whose
// output is 97% padding). The command's structure is exact; the quality the
// split costs is the teacher-forced KL gate.
constexpr double kMaximumError = 0.08;

int failures = 0;
int sectionFailures = 0;

void fail(const std::string &what) {
  constexpr int kShown = 8;
  if (sectionFailures++ < kShown) std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) {
  std::cout << what << (sectionFailures ? " FAIL (" + std::to_string(sectionFailures) + " failures)" : " ok")
            << '\n';
  sectionFailures = 0;
}

// The routing of the tests: a sigmoid router that selects by score plus a
// bias, no shared slices, and Inkling-2-Mini's softcaps.
MoeShape shape() {
  MoeShape shape;
  shape.hiddenSize = kHidden;
  shape.experts = kExperts;
  shape.expertsPerToken = kTopK;
  shape.expertIntermediateSize = kIntermediate;
  shape.router = MoeRouter::SigmoidNormalized;
  shape.sharedSlices = 0;
  shape.sharedGate = false;
  shape.selectionBias = true;
  shape.routedScale = 1.0f;
  shape.gateSoftcap = 4;
  shape.upSoftcap = 25;
  return shape;
}

// The corpus's routes of the block's experts: 1 and 3 the most-routed.
AneMoe::Loads loads() { return {{10, 90, 5, 70}}; }

// The block's expert `expert`'s FFN of the bf16 row `x`, as the ANE's group
// computes it: the softcapped SwiGLU of kernels/common/softcapped_swiglu.h,
// each projection's sum times its expert's scale, in double.
std::vector<double> forward(const moe_fixture::Model &model, uint32_t expert, const std::vector<float> &x) {
  const auto &gate = model.routed[0], &up = model.routed[1], &down = model.routed[2];
  const double gateScale = model.scale(0, expert), upScale = model.scale(1, expert),
               downScale = model.scale(2, expert);
  std::vector<double> h(kIntermediate);
  for (uint32_t i = 0; i < kIntermediate; ++i) {
    double g = 0.0, u = 0.0;
    const float *wg = gate.row(uint64_t{expert} * kIntermediate + i);
    const float *wu = up.row(uint64_t{expert} * kIntermediate + i);
    for (uint32_t k = 0; k < kHidden; ++k) {
      g += double(wg[k]) * double(x[k]);
      u += double(wu[k]) * double(x[k]);
    }
    g *= gateScale;
    u *= upScale;
    const double factor = 4.0 * std::tanh(g / 4.0) * (1.0 / (1.0 + std::exp(-g))),
                 value = 25.0 * std::tanh(u / 25.0);
    h[i] = factor * value;
  }
  std::vector<double> y(kHidden);
  for (uint32_t j = 0; j < kHidden; ++j) {
    const float *wd = down.row(uint64_t{expert} * kHidden + j);
    double sum = 0.0;
    for (uint32_t i = 0; i < kIntermediate; ++i) sum += double(wd[i]) * h[i];
    y[j] = sum * downScale;
  }
  return y;
}

void planChecks() {
  const MoeShape block = shape();
  const std::span<const MoeShape> shapes(&block, 1);
  const AneMoe::Loads corpus = loads();
  // A budget holds no group, then two experts, then every one.
  if (!AneMoe::select(shapes, corpus, 0).empty()) fail("a budget of no bytes selected a plan");
  AneMoe::Selection previous;
  for (const uint64_t budget : {AneMoe::expertBytes(block, 1), AneMoe::expertBytes(block, 2),
                                AneMoe::expertBytes(block, 3), AneMoe::expertBytes(block, 4), uint64_t{1} << 40}) {
    const AneMoe::Selection selected = AneMoe::select(shapes, corpus, budget);
    if (selected.empty()) {
      fail("a budget of " + std::to_string(budget) + " bytes selected no plan");
      continue;
    }
    if (selected.bytes > budget) fail("a plan takes more bytes than its budget");
    if (!selected.capacity || selected.capacity % AneMoe::kCapacityQuantum)
      fail("a plan's capacity is not whole quanta");
    if (uint64_t{selected.experts} * selected.capacity > AneMoe::kMaximumRows)
      fail("a plan's group holds more rows than a function");
    if (selected.covered < previous.covered - 1e-9) fail("a larger budget covers fewer routes");
    if (!previous.empty() && selected.bytes < previous.bytes) fail("a larger budget takes fewer bytes");
    previous = selected;
  }
  // The plan names each block's most-routed experts, the capacity it takes.
  const AneMoe::Selection two{.experts = kResidentExperts, .capacity = kCapacity};
  const AneMoe::Plan plan = AneMoe::plan(shapes, corpus, two);
  if (plan.resident.size() != 1 || plan.resident.front() != std::vector<uint32_t>{1, 3})
    fail("the plan's experts are not the most-routed");
  if (plan.rows() != kResidentExperts * kCapacity) fail("the plan's group rows are not its experts' capacities");
  AneMoe::Loads uniform{{1, 1, 1, 1}};
  if (AneMoe::plan(shapes, uniform, two).resident.front() != std::vector<uint32_t>{0, 1})
    fail("uniform loads take other experts than the first");
  // The plan's capacity and experts must be ones the blocks hold.
  splash::test::rejects([&] { static_cast<void>(AneMoe::plan(shapes, corpus, AneMoe::Selection{.experts = 5, .capacity = 32})); },
                        "more experts", "a plan of more experts than a block accepted");
  splash::test::rejects([&] { static_cast<void>(AneMoe::plan(shapes, corpus, AneMoe::Selection{.experts = 2, .capacity = 33})); },
                        "capacity", "a plan of a capacity not whole quanta accepted");
  // The key names the blocks' shapes and formats.
  const std::string key = AneMoe::calibrationKey(std::span<const BlockMoeWeights *const>{}, block);
  if (key.find("256x128e4") == std::string::npos) fail("the calibration key does not name the shapes: " + key);
}

// The chunk's routes to `selected`, and its bf16 input rows.
void fillRoutes(const metal::MetalBackend &, std::vector<uint32_t> &selected) {
  selected.assign(kRows * kTopK, 0);
  uint32_t route = 0;
  for (const uint32_t expert : {1u, 3u})
    for (uint32_t i = 0; i < kRoutesToResident; ++i) selected[route++] = expert;
  for (route = 2 * kRoutesToResident; route < selected.size(); ++route)
    selected[route] = route % 2 ? 0u : 2u;
}

void splitChecks(MetalBackend &backend) {
  const MoeShape block = shape();
  const std::span<const MoeShape> shapes(&block, 1);
  moe_fixture::Model model = moe_fixture::makeModel(backend, block, 0, false);
  const AneMoe::Plan plan =
      AneMoe::plan(shapes, loads(), AneMoe::Selection{.experts = kResidentExperts, .capacity = kCapacity});
  std::cout << "  block " << moe_fixture::formatNames(model) << ", experts " << plan.experts << " and "
            << plan.resident.front()[0] << "/" << plan.resident.front()[1] << " of capacity " << plan.capacity << '\n';
  const std::array<const BlockMoeWeights *, 1> blocks{&model.weights};
  AneMoe split(backend, std::span(blocks), block, plan, {});
  if (split.regionRows() != plan.rows()) fail("the block's region is not its group's rows");
  if (split.splits(AneMoe::kMinimumRows - 1) || !split.splits(AneMoe::kMinimumRows) ||
      !split.splits(kRows) || !split.splits(AneMoe::kMaximumRows) || split.splits(AneMoe::kMaximumRows + 1))
    fail("the split takes commands of other rows");

  // The chunk: its routes, its input rows, and the scratch the split writes.
  const MoePlan moe = MoE::plan(block, kRows, MoeConfig{});
  std::vector<uint32_t> selected, original;
  fillRoutes(backend, selected);
  original = selected;
  const uint64_t routes = uint64_t{kRows} * block.routesPerToken();
  auto buffer = [&](uint64_t bytes) { return test::sharedBuffer(backend, bytes); };
  MoeBuffers buffers;
  buffers.input = buffer(uint64_t{kRows} * kHidden * sizeof(uint16_t));
  buffers.scratch.selectedExperts = buffer(routes * sizeof(uint32_t));
  buffers.scratch.routeRows = buffer(routes * sizeof(uint32_t));
  buffers.scratch.expertOutput =
      buffer((routes + plan.rows()) * kHidden * sizeof(uint16_t));
  std::vector<float> input(uint64_t{kRows} * kHidden);
  for (uint32_t row = 0; row < kRows; ++row) {
    std::normal_distribution<float> normal;
    for (uint32_t k = 0; k < kHidden; ++k) input[uint64_t{row} * kHidden + k] = normal(moe_fixture::rng);
  }
  auto *rows = static_cast<__bf16 *>(buffers.input.contents());
  for (uint64_t i = 0; i < input.size(); ++i) rows[i] = __bf16(input[i]);
  const uint32_t kUntouchedRow = 0xFFFFFFFFu;

  // A command of the split's routes: the mark, the pack, the evaluation and
  // the join, in one command as the target encodes them.
  struct Run final {
    std::vector<uint16_t> output;
    std::vector<uint32_t> routeRows;
    bool usable = false;
  };
  const auto command = [&](Run &run) -> bool {
    std::memcpy(buffers.scratch.selectedExperts.contents(), selected.data(), selected.size() * sizeof(uint32_t));
    std::memset(buffers.scratch.routeRows.contents(), 0xFF, routes * sizeof(uint32_t));
    std::memset(buffers.scratch.expertOutput.contents(), 0, (routes + plan.rows()) * kHidden * sizeof(uint16_t));
    static_cast<void>(split.begin(kRows));
    CommandGraph graph;
    split.addRouting(graph, buffers, moe, model.weights);
    split.addJoin(graph, buffers, moe, model.weights);
    static_cast<void>(split.commit(graph, {}).wait());
    run.usable = split.finish();
    const auto *values = static_cast<const uint16_t *>(buffers.scratch.expertOutput.contents());
    run.output.assign(values, values + (routes + plan.rows()) * kHidden);
    const auto *rows = static_cast<const uint32_t *>(buffers.scratch.routeRows.contents());
    run.routeRows.assign(rows, rows + routes);
    return run.usable;
  };
  Run first, second;
  if (!command(first)) fail("the split's first command was not usable");
  if (!command(second)) fail("the split's second command was not usable");
  // The mark's packed columns depend on the order its threads claim them; a
  // route's joined rows do not.
  for (uint32_t route = 0; route < routes; ++route) {
    if (first.routeRows[route] == kUntouchedRow || second.routeRows[route] == kUntouchedRow) continue;
    const auto *a = first.output.data() + uint64_t{first.routeRows[route]} * kHidden;
    const auto *b = second.output.data() + uint64_t{second.routeRows[route]} * kHidden;
    if (std::memcmp(a, b, kHidden * sizeof(uint16_t))) {
      fail("route " + std::to_string(route) + " joined other rows its second command");
      break;
    }
  }

  // The selection: the residents' routes up to their capacity leave the GPU,
  // and the others stay.
  const auto *marked = static_cast<const uint32_t *>(buffers.scratch.selectedExperts.contents());
  const auto *routeRows = static_cast<const uint32_t *>(buffers.scratch.routeRows.contents());
  uint32_t resident = 0, columns = 0;
  std::vector<uint32_t> columnOf(routes, UINT32_MAX);
  for (uint32_t route = 0; route < routes; ++route) {
    const bool ane = original[route] == 1 || original[route] == 3;
    if (ane) {
      ++resident;
      if (marked[route] != ANE_MOE_ROUTE_NONE) fail("route " + std::to_string(route) + " of a resident expert stayed");
      if (routeRows[route] < routes || routeRows[route] >= routes + plan.rows())
        fail("route " + std::to_string(route) + " has no row in the block's region: " +
             std::to_string(routeRows[route]));
      else
        columnOf[route] = routeRows[route] - static_cast<uint32_t>(routes);
    } else if (marked[route] != original[route] || routeRows[route] != kUntouchedRow) {
      fail("route " + std::to_string(route) + " of another expert was marked");
    }
  }
  if (resident != 2 * kRoutesToResident) fail("the mark took other routes than the residents'");
  // Each expert's first columns hold its routes, and its others are padding.
  std::vector<bool> filled(plan.rows(), false);
  for (uint32_t route = 0; route < routes; ++route)
    if (columnOf[route] != UINT32_MAX) {
      if (filled[columnOf[route]]) fail("two routes join at column " + std::to_string(columnOf[route]));
      filled[columnOf[route]] = true;
      const uint32_t slot = original[route] == 1 ? 0 : 1;
      const uint32_t column = columnOf[route];
      if (column / kCapacity != slot) fail("a route joins outside its expert's columns");
      ++columns;
    }
  if (columns != resident) fail("the join's columns are not the routes'");
  // The joined rows: each route's expert's forward of its token's input row,
  // and zeros where no route holds the column.
  const auto *joined = static_cast<const uint16_t *>(buffers.scratch.expertOutput.contents());
  double worst = 0.0;
  for (uint32_t route = 0; route < routes; ++route) {
    const uint32_t column = columnOf[route];
    if (column == UINT32_MAX) continue;
    const uint32_t token = route / block.routesPerToken();
    std::vector<float> x(kHidden);
    for (uint32_t k = 0; k < kHidden; ++k) x[k] = bf16ToFloat(floatToBf16(input[uint64_t{token} * kHidden + k]));
    const std::vector<double> expected = forward(model, original[route], x);
    const uint16_t *row = joined + uint64_t{routes + column} * kHidden;
    double difference = 0.0, scale = 0.0;
    for (uint32_t j = 0; j < kHidden; ++j) {
      const double value = bf16ToFloat(row[j]);
      difference += (value - expected[j]) * (value - expected[j]);
      scale += expected[j] * expected[j];
    }
    const double error = std::sqrt(difference / scale);
    worst = std::max(worst, error);
    if (!(error <= kMaximumError))
      fail("route " + std::to_string(route) + " of expert " + std::to_string(original[route]) + " differs by " +
           std::to_string(100.0 * error) + "%");
  }
  std::cout << "  worst route error " << 100.0 * worst << "%\n";
  for (uint32_t column = 0; column < plan.rows(); ++column) {
    if (filled[column]) continue;
    const uint16_t *row = joined + uint64_t{routes + column} * kHidden;
    for (uint32_t j = 0; j < kHidden; ++j)
      if (bf16ToFloat(row[j]) != 0.0f) {
        fail("padding column " + std::to_string(column) + " holds a value");
        break;
      }
  }
  // A plan of no block takes no command.
  AneMoe empty(backend, std::span(blocks), block, AneMoe::Plan{}, {});
  if (empty.splits(kRows) || empty.regionRows()) fail("a split of no block takes a command");
}

// A whole MoE step of the block — the routing, the grouping, the expert
// passes and the combine — with and without the split: the engine's routes
// leave the GPU's grouping and join before the combine, so both compute the
// block's output, within the ANE's error of the routes it takes.
void stepChecks(MetalBackend &backend) {
  // The router's float projection writes whole groups of eight experts, so a
  // block of four runs no whole step.
  MoeShape block = shape();
  block.experts = kStepExperts;
  moe_fixture::Model model = moe_fixture::makeModel(backend, block, 0, false);
  const AneMoe::Loads corpus{{10, 90, 5, 70, 20, 30, 15, 25}};
  const AneMoe::Plan plan = AneMoe::plan(std::span(&block, 1), corpus,
                                         AneMoe::Selection{.experts = kResidentExperts, .capacity = kCapacity});
  if (plan.resident.front() != std::vector<uint32_t>{1, 3})
    fail("the step's resident experts are not the most-routed");
  const std::array<const BlockMoeWeights *, 1> blocks{&model.weights};
  AneMoe split(backend, std::span(blocks), block, plan, {});
  const MoePlan plainPlan = MoE::plan(block, kRows, MoeConfig{});
  const MoePlan splitPlan = MoE::plan(block, kRows, MoeConfig{.aneRows = plan.rows()});
  if (splitPlan.workspace().expertOutputBytes <= plainPlan.workspace().expertOutputBytes)
    fail("a step with the engine's region sizes no more expert output than without");
  const uint64_t hiddenBytes = uint64_t{kRows} * kHidden * sizeof(uint16_t);
  const MetalBuffer input = test::sharedBuffer(backend, hiddenBytes);
  const MetalBuffer residual = test::sharedBuffer(backend, hiddenBytes);
  std::normal_distribution<float> normal;
  auto *rows = static_cast<__bf16 *>(input.contents());
  for (uint64_t i = 0; i < uint64_t{kRows} * kHidden; ++i) rows[i] = __bf16(normal(moe_fixture::rng));
  std::memset(residual.contents(), 0, hiddenBytes);
  struct Run final {
    std::vector<uint16_t> output;
    std::vector<uint32_t> selected;
    bool usable = true;
  };
  const auto run = [&](const MoePlan &moe, AneMoe *ane) {
    Run out;
    MoeBuffers buffers;
    buffers.input = input;
    buffers.residual = residual;
    buffers.output = test::sharedBuffer(backend, hiddenBytes);
    moe_fixture::Buffers allocated;
    moe_fixture::allocate(backend, allocated, moe);
    buffers.scratch = allocated.moe.scratch;
    CommandGraph graph;
    if (ane) static_cast<void>(ane->begin(kRows));
    MoE::addRouting(graph, buffers, model.weights, moe, ane);
    MoE::addRouted(graph, buffers, model.weights, moe, nullptr, ane);
    if (ane) {
      static_cast<void>(ane->commit(graph, {}).wait());
      out.usable = ane->finish();
    } else {
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    }
    const auto *values = static_cast<const uint16_t *>(buffers.output.contents());
    out.output.assign(values, values + uint64_t{kRows} * kHidden);
    const auto *selected = static_cast<const uint32_t *>(buffers.scratch.selectedExperts.contents());
    out.selected.assign(selected, selected + uint64_t{kRows} * block.routesPerToken());
    return out;
  };
  const Run alone = run(plainPlan, nullptr);
  const Run engine = run(splitPlan, &split);
  if (!engine.usable) fail("a step with the split was not usable");
  const uint32_t ane = static_cast<uint32_t>(
      std::count(engine.selected.begin(), engine.selected.end(), ANE_MOE_ROUTE_NONE));
  const uint32_t gpu = static_cast<uint32_t>(
      std::count_if(engine.selected.begin(), engine.selected.end(), [](uint32_t e) { return e != ANE_MOE_ROUTE_NONE; }));
  const uint32_t whole = static_cast<uint32_t>(
      std::count_if(alone.selected.begin(), alone.selected.end(), [](uint32_t e) { return e != ANE_MOE_ROUTE_NONE; }));
  if (!ane) fail("the split took none of the step's routes");
  if (whole != alone.selected.size()) fail("the GPU alone left a route ungrouped");
  if (gpu + ane != alone.selected.size()) fail("the step's routes are not the engine's and the GPU's");
  double difference = 0.0, scale = 0.0;
  for (uint64_t i = 0; i < alone.output.size(); ++i) {
    const double a = bf16ToFloat(alone.output[i]), b = bf16ToFloat(engine.output[i]);
    difference += (b - a) * (b - a);
    scale += a * a;
  }
  // The mark's counts: every route of the step, its resident experts' with
  // the engine's, and the loads a next start plans from, which round-trip.
  const AneMoe::Loads &loads = split.loads();
  uint64_t counted = 0;
  for (uint32_t expert = 0; expert < block.experts; ++expert) {
    const auto expected = static_cast<uint64_t>(std::count(alone.selected.begin(), alone.selected.end(), expert));
    counted += loads.front()[expert];
    if (loads.front()[expert] != expected)
      fail("expert " + std::to_string(expert) + " counted " + std::to_string(loads.front()[expert]) + " routes, not " +
           std::to_string(expected));
  }
  if (counted != alone.selected.size()) fail("the counts are not the step's routes");
  if (AneMoe::parseLoads(AneMoe::encodeLoads(loads), 1, block.experts) != loads)
    fail("the loads do not round-trip through their line");
  if (AneMoe::parseLoads("ane-moe loads 1 2", 1, block.experts).has_value())
    fail("a short line parsed as loads");
  const double error = scale > 0.0 ? std::sqrt(difference / scale) : 0.0;
  std::cout << "  step: " << ane << " of " << alone.selected.size() << " routes on the engine, output differs by "
            << 100.0 * error << "%\n";
  if (!(error <= 0.08)) fail("a step with the split differs from the GPU's by " + std::to_string(100.0 * error) + "%");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ane-moe METALLIB\n";
    return 2;
  }
  try {
    planChecks();
    section("plan");
    MetalBackend backend(argv[1]);
    splitChecks(backend);
    section("split");
    stepChecks(backend);
    section("step");
  } catch (const std::exception &error) {
    std::cout << "FAIL " << error.what() << '\n';
    return 1;
  }
  if (failures) {
    std::cout << failures << " failures\n";
    return 1;
  }
  std::cout << "ane-moe: all checks passed\n";
  return 0;
}
