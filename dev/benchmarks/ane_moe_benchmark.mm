#include "AneMoePlan.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/abi/AneMoe.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::dev::ane_moe;

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;
using Element = ane::Surface::Element;

constexpr uint32_t kInklingHidden = 2304;
constexpr uint32_t kInklingExpert = 1152;
constexpr uint32_t kInklingExperts = 128;
constexpr uint32_t kInklingTopK = 8;
constexpr float kInt8Range = 127.0f;
constexpr float kHeadroom = 8.0f;

struct Dimensions final {
  uint32_t hidden = kInklingHidden;
  uint32_t intermediate = kInklingExpert;
  uint32_t experts = kInklingExperts;
  uint32_t topK = kInklingTopK;
};

struct Options final {
  uint32_t experts = kInklingExperts;
  uint32_t topK = kInklingTopK;
  uint32_t groupSize = 4;
  uint32_t maxCapacity = 512;
  uint32_t rows = 0;
  uint64_t seed = 41;
  double seconds = 10.0;
  uint32_t hidden = kInklingHidden;
  uint32_t intermediate = kInklingExpert;
  uint32_t minimumGraphRows = 512;
  bool verify = true;
  std::string routes;
};

struct ExpertWeights final {
  ane::Surface gate;
  ane::Surface up;
  ane::Surface down;
  ane::Surface gateScale;
  ane::Surface upScale;
  ane::Surface downScale;
};

struct InputBatch final {
  uint32_t rows = 0;
  uint32_t hidden = 0;
  std::vector<int8_t> codes;
  std::vector<_Float16> scales;
  std::vector<float> values;
};

struct Shape final {
  uint32_t groups = 0;
  uint32_t capacity = 0;
  bool operator<(const Shape &other) const noexcept {
    return std::tie(groups, capacity) < std::tie(other.groups, other.capacity);
  }
};

std::string tensorDims(uint32_t rows, uint32_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}

std::string tensor(const char *type, uint32_t rows, uint32_t width) {
  return std::string("tensor<") + type + ", " + tensorDims(rows, width) + ">";
}

std::string fp16(double value) {
  char text[48];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}

uint32_t elementBytes(Element element) { return element == Element::Int8 ? 1 : 2; }

uint32_t alignedRowBytes(uint32_t width, Element element) {
  const uint32_t bytes = width * elementBytes(element);
  return (bytes + 63) / 64 * 64;
}

std::string bufferType(Element element, uint32_t rows, uint32_t width) {
  const uint32_t stride = alignedRowBytes(width, element) / elementBytes(element);
  const uint64_t plane = uint64_t{rows} * stride;
  const char *type = element == Element::Int8 ? "int8" : "fp16";
  return std::string("tensor_buffer<") + type + ", shape=" + tensorDims(rows, width) + ", strides=[" +
         std::to_string(plane) + ", " + std::to_string(plane) + ", " + std::to_string(stride) +
         ", 1], interleave_factors=[1, 1, 1, 1]>";
}

std::string makeMil(uint32_t groupCount, uint32_t capacity, uint32_t hidden, uint32_t intermediate) {
  const uint32_t rows = groupCount * capacity;
  std::string parameters = bufferType(Element::Int8, hidden, rows) + " x, " +
                           bufferType(Element::Float16, 1, rows) + " tx";
  std::string body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };

  line(tensor("int8", hidden, rows) + " x_t = tensor_buffer_to_tensor<ios17>(input = x)");
  line(tensor("fp16", hidden, rows) + " x_unit = dequantize(input = x_t, scale = " + fp16(1.0) + ")");
  line(tensor("fp16", 1, rows) + " tx_t = tensor_buffer_to_tensor<ios17>(input = tx)");
  line(tensor("fp16", hidden, rows) + " x_d = mul(x = x_unit, y = tx_t)");

  std::string outputs;
  for (uint32_t expert = 0; expert < groupCount; ++expert) {
    const std::string tag = std::to_string(expert);
    const uint32_t begin = expert * capacity;
    parameters += ", " + bufferType(Element::Int8, intermediate, hidden) + " wg" + tag;
    parameters += ", " + bufferType(Element::Float16, intermediate, 1) + " sg" + tag;
    parameters += ", " + bufferType(Element::Int8, intermediate, hidden) + " wu" + tag;
    parameters += ", " + bufferType(Element::Float16, intermediate, 1) + " su" + tag;
    parameters += ", " + bufferType(Element::Int8, hidden, intermediate) + " wd" + tag;
    parameters += ", " + bufferType(Element::Float16, hidden, 1) + " sd" + tag;

    line(tensor("int8", intermediate, hidden) + " wg" + tag + "_t = tensor_buffer_to_tensor<ios17>(input = wg" + tag + ")");
    line(tensor("int8", intermediate, hidden) + " wu" + tag + "_t = tensor_buffer_to_tensor<ios17>(input = wu" + tag + ")");
    line(tensor("int8", hidden, intermediate) + " wd" + tag + "_t = tensor_buffer_to_tensor<ios17>(input = wd" + tag + ")");
    line(tensor("fp16", intermediate, hidden) + " wg" + tag + "_d = dequantize(input = wg" + tag + "_t, scale = " +
         fp16(1.0 / kInt8Range) + ")");
    line(tensor("fp16", intermediate, hidden) + " wu" + tag + "_d = dequantize(input = wu" + tag + "_t, scale = " +
         fp16(1.0 / kInt8Range) + ")");
    line(tensor("fp16", hidden, intermediate) + " wd" + tag + "_d = dequantize(input = wd" + tag + "_t, scale = " +
         fp16(1.0 / kInt8Range) + ")");
    line(tensor("fp16", intermediate, 1) + " sg" + tag + "_t = tensor_buffer_to_tensor<ios17>(input = sg" + tag + ")");
    line(tensor("fp16", intermediate, 1) + " su" + tag + "_t = tensor_buffer_to_tensor<ios17>(input = su" + tag + ")");
    line(tensor("fp16", hidden, 1) + " sd" + tag + "_t = tensor_buffer_to_tensor<ios17>(input = sd" + tag + ")");
    const std::string inputX = groupCount > 1 ? "x" + tag : "x_d";
    if (groupCount > 1)
      line(tensor("fp16", hidden, capacity) + " " + inputX +
           " = slice_by_size(x = x_d, begin = tensor<int32, [4]>([0, 0, 0, " + std::to_string(begin) +
           "]), size = tensor<int32, [4]>([1, 1, " + std::to_string(hidden) + ", " + std::to_string(capacity) + "]))");

    line(tensor("fp16", intermediate, capacity) + " g0" + tag +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wg" + tag + "_d, y = " + inputX + ")");
    line(tensor("fp16", intermediate, capacity) + " g" + tag + " = mul(x = g0" + tag + ", y = sg" + tag + "_t)");
    line(tensor("fp16", intermediate, capacity) + " u0" + tag +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wu" + tag + "_d, y = " + inputX + ")");
    line(tensor("fp16", intermediate, capacity) + " u" + tag + " = mul(x = u0" + tag + ", y = su" + tag + "_t)");

    line(tensor("fp16", intermediate, capacity) + " g4" + tag + " = mul(x = g" + tag + ", y = " + fp16(0.25) + ")");
    line(tensor("fp16", intermediate, capacity) + " g4t" + tag + " = tanh(x = g4" + tag + ")");
    line(tensor("fp16", intermediate, capacity) + " gcap" + tag + " = mul(x = g4t" + tag + ", y = " + fp16(4.0) + ")");
    line(tensor("fp16", intermediate, capacity) + " g2" + tag + " = mul(x = g" + tag + ", y = " + fp16(0.5) + ")");
    line(tensor("fp16", intermediate, capacity) + " g2t" + tag + " = tanh(x = g2" + tag + ")");
    line(tensor("fp16", intermediate, capacity) + " g2p" + tag + " = add(x = g2t" + tag + ", y = " + fp16(1.0) + ")");
    line(tensor("fp16", intermediate, capacity) + " gs" + tag + " = mul(x = g2p" + tag + ", y = " + fp16(0.5) + ")");
    line(tensor("fp16", intermediate, capacity) + " gate" + tag + " = mul(x = gcap" + tag + ", y = gs" + tag + ")");
    line(tensor("fp16", intermediate, capacity) + " u25" + tag + " = mul(x = u" + tag + ", y = " + fp16(1.0 / 25.0) + ")");
    line(tensor("fp16", intermediate, capacity) + " u25t" + tag + " = tanh(x = u25" + tag + ")");
    line(tensor("fp16", intermediate, capacity) + " ucap" + tag + " = mul(x = u25t" + tag + ", y = " + fp16(25.0) + ")");
    line(tensor("fp16", intermediate, capacity) + " h0" + tag + " = mul(x = gate" + tag + ", y = ucap" + tag + ")");
    line(tensor("fp16", intermediate, capacity) + " h" + tag + " = mul(x = h0" + tag + ", y = " +
         fp16(1.0 / kHeadroom) + ")");
    line(tensor("fp16", hidden, capacity) + " y0" + tag +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wd" + tag + "_d, y = h" + tag + ")");
    line(tensor("fp16", hidden, capacity) + " y" + tag + " = mul(x = y0" + tag + ", y = sd" + tag + "_t)");
    if (!outputs.empty()) outputs += ", ";
    outputs += "y" + tag;
  }

  std::string output = "y0";
  if (groupCount > 1) {
    output = "yall";
    line(tensor("fp16", hidden, rows) + " yall = concat(axis = int32(3), interleave = bool(false), values = (" +
         outputs + "))");
  }
  line(bufferType(Element::Float16, hidden, rows) + " ybuf = tensor_to_tensor_buffer<ios17>(input = " + output +
       ", interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>([" +
       std::to_string(uint64_t{hidden} * (alignedRowBytes(rows, Element::Float16) / 2)) + ", " +
       std::to_string(uint64_t{hidden} * (alignedRowBytes(rows, Element::Float16) / 2)) + ", " +
       std::to_string(alignedRowBytes(rows, Element::Float16) / 2) + ", 1]))");
  return "program(1.3)\n{\n    func ane_moe<ios18>(" + parameters + ") {\n" + body + "    } -> (ybuf);\n}\n";
}

Chunk makeRoutes(uint32_t rows, uint32_t experts, uint32_t topK, uint64_t seed, double skew = 0.8) {
  if (topK > experts) throw std::invalid_argument("top-k exceeds expert count");
  std::mt19937_64 random(seed);
  std::normal_distribution<float> score(0.0f, 1.0f);
  std::vector<double> popularity(experts);
  for (uint32_t expert = 0; expert < experts; ++expert)
    popularity[expert] = 1.0 / std::pow(double(expert + 1), skew);
  std::discrete_distribution<uint32_t> choose(popularity.begin(), popularity.end());

  Chunk chunk;
  chunk.rows = rows;
  chunk.experts = experts;
  chunk.topK = topK;
  chunk.routes.reserve(uint64_t{rows} * topK);
  std::vector<uint32_t> selected;
  selected.reserve(topK);
  std::vector<float> raw(topK);
  for (uint32_t token = 0; token < rows; ++token) {
    selected.clear();
    while (selected.size() < topK) {
      const uint32_t expert = choose(random);
      if (std::ranges::find(selected, expert) == selected.end()) selected.push_back(expert);
    }
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < topK; ++slot) {
      raw[slot] = 1.0f / (1.0f + std::exp(-score(random)));
      sum += raw[slot];
    }
    for (uint32_t slot = 0; slot < topK; ++slot)
      chunk.routes.push_back({token, selected[slot], raw[slot] * float(topK) / sum});
  }
  chunk.validate();
  return chunk;
}

InputBatch makeInput(uint32_t rows, uint32_t hidden, uint64_t seed) {
  InputBatch input;
  input.rows = rows;
  input.hidden = hidden;
  input.codes.resize(uint64_t{rows} * hidden);
  input.scales.resize(rows);
  input.values.resize(uint64_t{rows} * hidden);
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (uint32_t token = 0; token < rows; ++token) {
    std::vector<float> values(hidden);
    float maximum = 0.0f;
    for (float &value : values) {
      value = normal(random);
      maximum = std::max(maximum, std::abs(value));
    }
    const float quantScale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    input.scales[token] = _Float16(quantScale);
    const float effectiveScale = static_cast<float>(input.scales[token]);
    for (uint32_t feature = 0; feature < hidden; ++feature) {
      const long quantized = std::lround(values[feature] / quantScale);
      const int8_t code = static_cast<int8_t>(std::clamp<long>(quantized, -127, 127));
      const uint64_t index = uint64_t{token} * hidden + feature;
      input.codes[index] = code;
      input.values[index] = float(code) * effectiveScale;
    }
  }
  return input;
}

void fillProjection(ane::Surface &weights, ane::Surface &scales, uint32_t outputs, uint32_t inputs,
                    std::mt19937_64 &random, float scaleMultiplier = 1.0f) {
  auto *weightBytes = static_cast<uint8_t *>(weights.buffer.contents());
  auto *scaleValues = static_cast<_Float16 *>(scales.buffer.contents());
  std::normal_distribution<float> normal(0.0f, 1.0f / std::sqrt(float(inputs)));
  std::vector<float> row(inputs);
  for (uint32_t output = 0; output < outputs; ++output) {
    float maximum = 0.0f;
    for (float &value : row) {
      value = normal(random);
      maximum = std::max(maximum, std::abs(value));
    }
    const float quantScale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    scaleValues[uint64_t{output} * (scales.strideBytes / 2)] = _Float16(quantScale * scaleMultiplier);
    int8_t *destination = reinterpret_cast<int8_t *>(weightBytes + uint64_t{output} * weights.strideBytes);
    for (uint32_t input = 0; input < inputs; ++input) {
      destination[input] = static_cast<int8_t>(std::clamp<long>(std::lround(row[input] / quantScale), -127, 127));
    }
  }
}

std::vector<ExpertWeights> makeWeights(metal::MetalBackend &backend, const Dimensions &dims, uint64_t seed) {
  std::vector<ExpertWeights> experts;
  experts.reserve(dims.experts);
  std::mt19937_64 random(seed);
  for (uint32_t expert = 0; expert < dims.experts; ++expert) {
    ExpertWeights weights;
    weights.gate = ane::Surface::create(backend, dims.intermediate, dims.hidden, Element::Int8);
    weights.up = ane::Surface::create(backend, dims.intermediate, dims.hidden, Element::Int8);
    weights.down = ane::Surface::create(backend, dims.hidden, dims.intermediate, Element::Int8);
    weights.gateScale = ane::Surface::create(backend, dims.intermediate, 1, Element::Float16);
    weights.upScale = ane::Surface::create(backend, dims.intermediate, 1, Element::Float16);
    weights.downScale = ane::Surface::create(backend, dims.hidden, 1, Element::Float16);
    fillProjection(weights.gate, weights.gateScale, dims.intermediate, dims.hidden, random, kInt8Range);
    fillProjection(weights.up, weights.upScale, dims.intermediate, dims.hidden, random, kInt8Range);
    fillProjection(weights.down, weights.downScale, dims.hidden, dims.intermediate, random, kHeadroom * kInt8Range);
    experts.push_back(std::move(weights));
  }
  return experts;
}

struct GraphCompletion final {
  std::mutex mutex;
  std::condition_variable condition;
  bool completed = false;
  bool succeeded = false;
};

class Graph final {
public:
  Graph(uint32_t groups, uint32_t capacity, const Dimensions &dims)
      : program_(makeMil(groups, capacity, dims.hidden, dims.intermediate), {},
                 ane::Program::Limits{std::chrono::seconds(600), nullptr},
                 ane::cacheDirectory() / "moe-prototype") {
    procedure_ = program_.procedure("ane_moe");
  }

  // The compiled model may list its inputs in another order than the MIL
  // declares them, so surfaces are bound by name as AneFfn binds them.
  [[nodiscard]] ane::Program::Binding bind(const std::map<std::string, ane::Surface> &named,
                                          const ane::Surface &output) const {
    std::vector<ane::Surface> inputs;
    for (const std::string &name : program_.inputs(procedure_)) {
      const auto found = named.find(name);
      if (found == named.end()) throw std::logic_error("ANE MoE program has an unknown input " + name);
      inputs.push_back(found->second);
    }
    return program_.bind(procedure_, inputs, output);
  }

  // One evaluation queued that starts when `event` reaches `wait` and signals
  // `signal` when done: how a command's join waits for it (AneFfn's handoff).
  std::shared_ptr<GraphCompletion> enqueue(const ane::Program::Binding &binding, metal::SharedEvent &event,
                                          uint64_t wait, uint64_t signal) {
    const auto completion = std::make_shared<GraphCompletion>();
    program_.enqueue(binding, event, wait, signal, [completion](bool success) {
      {
        std::lock_guard lock(completion->mutex);
        completion->succeeded = success;
        completion->completed = true;
      }
      completion->condition.notify_one();
    });
    return completion;
  }
  // One evaluation queued after the caller's own: await() it before reading
  // its output.
  std::shared_ptr<GraphCompletion> enqueue(const ane::Program::Binding &binding, metal::SharedEvent &event,
                                          uint64_t &eventValue) {
    const uint64_t signal = ++eventValue;
    return enqueue(binding, event, signal - 1, signal);
  }
  static void await(const std::shared_ptr<GraphCompletion> &completion) {
    std::unique_lock lock(completion->mutex);
    if (!completion->condition.wait_for(lock, std::chrono::seconds(600), [&] { return completion->completed; }))
      throw std::runtime_error("ANE evaluation did not report completion within 600 seconds");
    if (!completion->succeeded) throw std::runtime_error("ANE evaluation failed");
  }
  void evaluate(const ane::Program::Binding &binding, metal::SharedEvent &event, uint64_t &eventValue) {
    await(enqueue(binding, event, eventValue));
  }

private:
  ane::Program program_;
  uint32_t procedure_ = 0;
};

class GraphCache final {
public:
  explicit GraphCache(const Dimensions &dims) : dims_(dims) {}

  Graph &get(uint32_t groups, uint32_t capacity) {
    const Shape shape{groups, capacity};
    auto found = graphs_.find(shape);
    if (found == graphs_.end())
      found = graphs_.emplace(shape, std::make_unique<Graph>(groups, capacity, dims_)).first;
    return *found->second;
  }

private:
  const Dimensions &dims_;
  std::map<Shape, std::unique_ptr<Graph>> graphs_;
};

struct GroupWork final {
  ane::Surface x;
  ane::Surface tx;
  ane::Surface y;
  std::optional<ane::Program::Binding> binding;
};

// A group's ANE surfaces, empty: the layer fills them on the CPU, the
// pipeline's GPU kernels write them.
GroupWork prepareSurfaces(metal::MetalBackend &backend, const PackedGroup &group, uint32_t hidden) {
  const uint32_t groupRows = group.capacity * static_cast<uint32_t>(group.experts.size());
  GroupWork work;
  work.x = ane::Surface::create(backend, hidden, groupRows, Element::Int8);
  work.tx = ane::Surface::create(backend, 1, groupRows, Element::Float16);
  work.y = ane::Surface::create(backend, hidden, groupRows, Element::Float16);
  std::memset(work.x.buffer.contents(), 0, ane::Surface::bytes(hidden, groupRows, Element::Int8));
  std::memset(work.tx.buffer.contents(), 0, ane::Surface::bytes(1, groupRows, Element::Float16));
  return work;
}

// The group's weights bound into its graph over `work`'s surfaces.
void bindGroup(Graph &graph, GroupWork &work, const PackedGroup &group, const std::vector<ExpertWeights> &weights) {
  std::map<std::string, ane::Surface> surfaces{{"x", work.x}, {"tx", work.tx}};
  for (uint32_t slot = 0; slot < group.experts.size(); ++slot) {
    const ExpertWeights &item = weights.at(group.experts[slot]);
    const std::string tag = std::to_string(slot);
    surfaces.emplace("wg" + tag, item.gate);
    surfaces.emplace("sg" + tag, item.gateScale);
    surfaces.emplace("wu" + tag, item.up);
    surfaces.emplace("su" + tag, item.upScale);
    surfaces.emplace("wd" + tag, item.down);
    surfaces.emplace("sd" + tag, item.downScale);
  }
  work.binding = graph.bind(surfaces, work.y);
}

GroupWork prepareGroup(metal::MetalBackend &backend, Graph &graph, const PackedGroup &group,
                       const InputBatch &input, const std::vector<ExpertWeights> &weights,
                       uint32_t hidden) {
  GroupWork work = prepareSurfaces(backend, group, hidden);
  const uint32_t groupRows = group.capacity * static_cast<uint32_t>(group.experts.size());
  auto *x = static_cast<uint8_t *>(work.x.buffer.contents());
  auto *tx = static_cast<_Float16 *>(work.tx.buffer.contents());
  std::fill_n(tx, groupRows, _Float16(1.0f));
  for (uint32_t column = 0; column < group.rows.size(); ++column) {
    const PackedRoute &route = group.rows[column];
    if (route.routeIndex == UINT32_MAX) continue;
    tx[column] = input.scales.at(route.token);
    for (uint32_t feature = 0; feature < hidden; ++feature) {
      const uint64_t source = uint64_t{route.token} * hidden + feature;
      x[uint64_t{feature} * work.x.strideBytes + column] = static_cast<uint8_t>(input.codes[source]);
    }
  }
  bindGroup(graph, work, group, weights);
  return work;
}

float scaleAt(const ane::Surface &scales, uint32_t row, float correction = 1.0f) {
  const auto *values = static_cast<const _Float16 *>(scales.buffer.contents());
  return static_cast<float>(values[uint64_t{row} * (scales.strideBytes / 2)]) * correction;
}

int8_t weightAt(const ane::Surface &weights, uint32_t row, uint32_t column) {
  const auto *bytes = static_cast<const uint8_t *>(weights.buffer.contents());
  return reinterpret_cast<const int8_t *>(bytes + uint64_t{row} * weights.strideBytes)[column];
}

std::vector<float> expertForward(const ExpertWeights &weights, const InputBatch &input, uint32_t token,
                                 const Dimensions &dims) {
  std::vector<float> gate(dims.intermediate);
  std::vector<float> up(dims.intermediate);
  for (uint32_t channel = 0; channel < dims.intermediate; ++channel) {
    float gateSum = 0.0f;
    float upSum = 0.0f;
    for (uint32_t feature = 0; feature < dims.hidden; ++feature) {
      const float x = input.values[uint64_t{token} * dims.hidden + feature];
      gateSum += float(weightAt(weights.gate, channel, feature)) * x;
      upSum += float(weightAt(weights.up, channel, feature)) * x;
    }
    gate[channel] = gateSum * scaleAt(weights.gateScale, channel, 1.0f / kInt8Range);
    up[channel] = upSum * scaleAt(weights.upScale, channel, 1.0f / kInt8Range);
  }

  std::vector<float> result(dims.hidden);
  for (uint32_t feature = 0; feature < dims.hidden; ++feature) {
    float sum = 0.0f;
    for (uint32_t channel = 0; channel < dims.intermediate; ++channel) {
      const float g = gate[channel];
      const float u = up[channel];
      const float activation = 4.0f * std::tanh(g / 4.0f) * (1.0f / (1.0f + std::exp(-g))) *
                               (25.0f * std::tanh(u / 25.0f));
      sum += float(weightAt(weights.down, feature, channel)) * activation;
    }
    result[feature] = sum * scaleAt(weights.downScale, feature, 1.0f / (kHeadroom * kInt8Range));
  }
  return result;
}

struct LayerResult final {
  std::vector<float> output;
  double packingMs = 0.0;
  double aneMs = 0.0;
  double fallbackMs = 0.0;
  double mergeMs = 0.0;
  PackStats stats;
};

LayerResult runLayer(metal::MetalBackend &backend, GraphCache &graphs, const Planner &planner, const Chunk &routes,
                     const InputBatch &input, const std::vector<ExpertWeights> &weights, const Dimensions &dims,
                     metal::SharedEvent &event, uint64_t &eventValue, bool computeFallback = true) {
  LayerResult result;
  result.output.assign(uint64_t{routes.rows} * dims.hidden, 0.0f);
  const auto packingStart = Clock::now();
  const Packed packed = planner.pack(routes);
  result.stats = packed.stats;
  const double plannerPackingMs = Seconds(Clock::now() - packingStart).count() * 1e3;
  result.packingMs += plannerPackingMs;

  for (const PackedGroup &group : packed.groups) {
    Graph &graph = graphs.get(static_cast<uint32_t>(group.experts.size()), group.capacity);
    const auto prepareStart = Clock::now();
    GroupWork work = prepareGroup(backend, graph, group, input, weights, dims.hidden);
    result.packingMs += Seconds(Clock::now() - prepareStart).count() * 1e3;
    const auto aneStart = Clock::now();
    graph.evaluate(*work.binding, event, eventValue);
    result.aneMs += Seconds(Clock::now() - aneStart).count() * 1e3;

    const auto mergeStart = Clock::now();
    const auto *output = static_cast<const _Float16 *>(work.y.buffer.contents());
    for (uint32_t column = 0; column < group.rows.size(); ++column) {
      const PackedRoute &route = group.rows[column];
      if (route.routeIndex == UINT32_MAX) continue;
      const float weight = routes.routes[route.routeIndex].weight;
      for (uint32_t feature = 0; feature < dims.hidden; ++feature) {
        const uint64_t index = uint64_t{feature} * (work.y.strideBytes / 2) + column;
        result.output[uint64_t{route.token} * dims.hidden + feature] += float(output[index]) * weight;
      }
    }
    result.mergeMs += Seconds(Clock::now() - mergeStart).count() * 1e3;
  }

  // The fallback stands in for the GPU's exact MoE path; timings at Inkling
  // dimensions count its routes rather than run scalar fp32 code for them.
  if (!computeFallback) return result;
  const auto fallbackStart = Clock::now();
  for (uint32_t routeIndex : packed.fallbackRoutes) {
    const Route &route = routes.routes[routeIndex];
    const std::vector<float> expert = expertForward(weights.at(route.expert), input, route.token, dims);
    for (uint32_t feature = 0; feature < dims.hidden; ++feature)
      result.output[uint64_t{route.token} * dims.hidden + feature] += expert[feature] * route.weight;
  }
  result.fallbackMs = Seconds(Clock::now() - fallbackStart).count() * 1e3;
  return result;
}

std::vector<float> referenceLayer(const Chunk &routes, const InputBatch &input,
                                  const std::vector<ExpertWeights> &weights, const Dimensions &dims) {
  std::vector<float> output(uint64_t{routes.rows} * dims.hidden, 0.0f);
  for (const Route &route : routes.routes) {
    const std::vector<float> expert = expertForward(weights.at(route.expert), input, route.token, dims);
    for (uint32_t feature = 0; feature < dims.hidden; ++feature)
      output[uint64_t{route.token} * dims.hidden + feature] += expert[feature] * route.weight;
  }
  return output;
}

std::pair<double, double> errorMetrics(std::span<const float> actual, std::span<const float> expected) {
  if (actual.size() != expected.size()) throw std::invalid_argument("numeric result sizes differ");
  // Relative RMS error, and the largest absolute error relative to the
  // reference's RMS (an element-wise ratio is dominated by outputs near zero).
  double squaredError = 0.0;
  double squaredExpected = 0.0;
  double maxError = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double difference = double(actual[i]) - expected[i];
    squaredError += difference * difference;
    squaredExpected += double(expected[i]) * expected[i];
    maxError = std::max(maxError, std::abs(difference));
  }
  const double expectedRms = std::sqrt(squaredExpected / std::max<size_t>(1, actual.size()));
  return {std::sqrt(squaredError / std::max(1e-30, squaredExpected)), maxError / std::max(1e-30, expectedRms)};
}

void planCheck() {
  std::vector<Chunk> calibration;
  for (int batch = 0; batch < 2; ++batch) {
    Chunk chunk;
    chunk.rows = 8;
    chunk.experts = 4;
    chunk.topK = 1;
    for (uint32_t token = 0; token < chunk.rows; ++token)
      chunk.routes.push_back({token, 0, 1.0f});
    static_cast<void>(batch);
    calibration.push_back(std::move(chunk));
  }
  const Planner planner = Planner::calibrate(calibration, Config{2, 32, 32, 0.95, 0});
  Chunk evaluation;
  evaluation.rows = 64;
  evaluation.experts = 4;
  evaluation.topK = 1;
  for (uint32_t token = 0; token < evaluation.rows; ++token)
    evaluation.routes.push_back({token, token < 40 ? 0u : 3u, 1.0f});
  const Packed packed = planner.pack(evaluation);
  if (!packed.coversExactly(evaluation) || planner.groups().size() != 1 ||
      packed.stats.aneAssignments != 32 || packed.stats.fallbackAssignments != 32 ||
      packed.stats.paddedRows != 0)
    throw std::runtime_error("planner invariant check failed");

  Chunk underfilled;
  underfilled.rows = 16;
  underfilled.experts = 4;
  underfilled.topK = 1;
  for (uint32_t token = 0; token < underfilled.rows; ++token) underfilled.routes.push_back({token, 0, 1.0f});
  const Packed underfilledPacked = planner.pack(underfilled);
  if (!underfilledPacked.coversExactly(underfilled) || underfilledPacked.stats.paddedRows != 16 ||
      underfilledPacked.stats.fallbackAssignments != 0)
    throw std::runtime_error("planner padding check failed");

  bool duplicateRejected = false;
  Chunk duplicate = evaluation;
  duplicate.topK = 2;
  duplicate.routes.clear();
  for (uint32_t token = 0; token < duplicate.rows; ++token) {
    duplicate.routes.push_back({token, 0, 0.5f});
    duplicate.routes.push_back({token, 0, 0.5f});
  }
  try {
    duplicate.validate();
  } catch (const std::invalid_argument &) {
    duplicateRejected = true;
  }
  if (!duplicateRejected) throw std::runtime_error("planner accepted duplicate expert assignments");
  std::printf("plan-check: groups=%zu capacity=%u overflow-useful=%llu overflow-fallback=%llu underfilled-padding=%llu; coverage and duplicate checks passed\n",
              planner.groups().size(), planner.groups().front().capacity,
              static_cast<unsigned long long>(packed.stats.aneAssignments),
              static_cast<unsigned long long>(packed.stats.fallbackAssignments),
              static_cast<unsigned long long>(underfilledPacked.stats.paddedRows));
}

void check(metal::MetalBackend &backend, const Options &options) {
  const Dimensions dims{options.hidden, options.intermediate, 16, 4};
  std::vector<Chunk> calibration{makeRoutes(64, dims.experts, dims.topK, 1001),
                                 makeRoutes(64, dims.experts, dims.topK, 2003)};
  // Longer than the calibration chunks, so hot experts overflow their capacity
  // and the exact fallback carries their remaining routes.
  const Chunk routes = makeRoutes(options.rows ? options.rows : 1024, dims.experts, dims.topK, 3001);
  const Planner planner = Planner::calibrate(calibration,
                                              Config{options.groupSize, 32, options.maxCapacity, 0.95, options.minimumGraphRows});
  const InputBatch input = makeInput(routes.rows, dims.hidden, 991);
  const std::vector<ExpertWeights> weights = makeWeights(backend, dims, 717);
  GraphCache graphs(dims);
  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t eventValue = 0;
  const LayerResult actual = runLayer(backend, graphs, planner, routes, input, weights, dims, event, eventValue);
  const auto referenceStart = Clock::now();
  const std::vector<float> expected = referenceLayer(routes, input, weights, dims);
  const double referenceMs = Seconds(Clock::now() - referenceStart).count() * 1e3;
  const auto [relativeRms, maxRelative] = errorMetrics(actual.output, expected);
  if (!actual.stats.assignments || actual.stats.aneAssignments + actual.stats.fallbackAssignments !=
                                      actual.stats.assignments || relativeRms > 0.03)
    throw std::runtime_error("ANE MoE numeric validation failed");
  std::printf("check: H=%u I=%u E=%u topK=%u rows=%u groups=%zu capacityRows=%llu useful=%llu fallback=%llu padding=%.1f%%\n",
              dims.hidden, dims.intermediate, dims.experts, dims.topK, routes.rows, planner.groups().size(),
              static_cast<unsigned long long>(actual.stats.capacityRows),
              static_cast<unsigned long long>(actual.stats.aneAssignments),
              static_cast<unsigned long long>(actual.stats.fallbackAssignments),
              actual.stats.capacityRows ? 100.0 * actual.stats.paddedRows / actual.stats.capacityRows : 0.0);
  std::printf("check: relative RMS error=%.5f max error=%.3f x reference RMS; pack %.2f ms, ANE %.2f ms, CPU fallback %.2f ms, merge %.2f ms, fp32 reference %.2f ms\n",
              relativeRms, maxRelative, actual.packingMs, actual.aneMs, actual.fallbackMs, actual.mergeMs, referenceMs);
}

void sweep(metal::MetalBackend &backend) {
  const Dimensions dims{kInklingHidden, kInklingExpert, 4, 1};
  const std::vector<ExpertWeights> weights = makeWeights(backend, dims, 131);
  GraphCache graphs(dims);
  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t eventValue = 0;
  std::printf("sweep: Inkling dimensions H=%u I=%u; timings include IOSurface binding and ANE evaluation, not weight generation\n",
              dims.hidden, dims.intermediate);
  // The floor: which group shapes the ANE compiles and runs at all. Production
  // AneFfn never goes below 512 rows (kMinimumRows), so smaller graphs are
  // measured here rather than assumed.
  for (uint32_t groups : {1u, 2u, 4u}) {
    for (uint32_t capacity : {32u, 64u, 128u, 256u, 512u, 1024u}) {
      const uint32_t rows = groups * capacity;
      try {
        const InputBatch input = makeInput(rows, dims.hidden, uint64_t{groups} * 1000 + capacity);
        PackedGroup group;
        group.capacity = capacity;
        for (uint32_t slot = 0; slot < groups; ++slot) {
          group.experts.push_back(slot);
          for (uint32_t row = 0; row < capacity; ++row) {
            const uint32_t token = slot * capacity + row;
            group.rows.push_back({token, token, slot, 0, row});
          }
        }
        Graph &graph = graphs.get(groups, capacity);
        double bestMs = 1e30;
        for (int repeat = 0; repeat < 4; ++repeat) {
          GroupWork work = prepareGroup(backend, graph, group, input, weights, dims.hidden);
          const auto start = Clock::now();
          graph.evaluate(*work.binding, event, eventValue);
          const double elapsedMs = Seconds(Clock::now() - start).count() * 1e3;
          if (repeat) bestMs = std::min(bestMs, elapsedMs);
        }
        const double flops = 2.0 * groups * capacity * dims.hidden * dims.intermediate * 3.0;
        std::printf("sweep: G=%u C=%u rows=%u best=%.2f ms useful=100%% padded=0%% throughput=%.3f TFLOP/s\n",
                    groups, capacity, rows, bestMs, flops / (bestMs * 1e9));
      } catch (const std::exception &error) {
        std::printf("sweep: G=%u C=%u rows=%u REJECTED %s\n", groups, capacity, rows, error.what());
      }
    }
  }
}

// Keeps one bound group graph evaluating back to back for --seconds, so the
// ANE stays busy while another process measures the GPU beside it.
void stress(metal::MetalBackend &backend, const Options &options) {
  const uint32_t groups = options.groupSize;
  const uint32_t rows = options.rows ? options.rows : 512;
  if (rows % groups || (rows / groups) % 32) throw std::invalid_argument("stress rows must split into multiple-of-32 capacities");
  const uint32_t capacity = rows / groups;
  const Dimensions dims{kInklingHidden, kInklingExpert, groups, 1};
  const std::vector<ExpertWeights> weights = makeWeights(backend, dims, 131);
  GraphCache graphs(dims);
  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t eventValue = 0;
  const InputBatch input = makeInput(rows, dims.hidden, 7);
  PackedGroup group;
  group.capacity = capacity;
  for (uint32_t slot = 0; slot < groups; ++slot) {
    group.experts.push_back(slot);
    for (uint32_t row = 0; row < capacity; ++row) {
      const uint32_t token = slot * capacity + row;
      group.rows.push_back({token, token, slot, 0, row});
    }
  }
  Graph &graph = graphs.get(groups, capacity);
  GroupWork work = prepareGroup(backend, graph, group, input, weights, dims.hidden);
  graph.evaluate(*work.binding, event, eventValue);
  const double flops = 2.0 * rows * dims.hidden * dims.intermediate * 3.0;
  const auto begin = Clock::now();
  auto windowStart = begin;
  uint64_t total = 0, window = 0;
  while (Seconds(Clock::now() - begin).count() < options.seconds) {
    graph.evaluate(*work.binding, event, eventValue);
    ++total;
    ++window;
    const double windowSeconds = Seconds(Clock::now() - windowStart).count();
    if (windowSeconds >= 1.0) {
      std::printf("stress: G=%u C=%u %.2f ms/eval %.3f TFLOP/s\n", groups, capacity, windowSeconds * 1e3 / window,
                  flops * window / (windowSeconds * 1e12));
      std::fflush(stdout);
      windowStart = Clock::now();
      window = 0;
    }
  }
  const double elapsed = Seconds(Clock::now() - begin).count();
  std::printf("stress: total G=%u C=%u evals=%llu %.2f ms/eval %.3f TFLOP/s\n", groups, capacity,
              static_cast<unsigned long long>(total), elapsed * 1e3 / total, flops * total / (elapsed * 1e12));
}

void layer(metal::MetalBackend &backend, const Options &options) {
  Dimensions dims{kInklingHidden, kInklingExpert, options.experts, options.topK};
  if (dims.topK > dims.experts) throw std::invalid_argument("top-k exceeds expert count");
  std::vector<Chunk> calibration;
  std::vector<Chunk> evaluations;
  if (!options.routes.empty()) {
    std::vector<Chunk> chunks = loadRoutes(options.routes, dims.experts, dims.topK);
    if (chunks.size() < 2) throw std::invalid_argument("route-file mode needs at least two chunks (calibration and held-out evaluation)");
    calibration.assign(chunks.begin(), chunks.end() - 1);
    evaluations.push_back(std::move(chunks.back()));
  } else {
    const std::vector<uint32_t> rowSizes = options.rows ? std::vector<uint32_t>{options.rows}
                                                        : std::vector<uint32_t>{512, 1024, 2048};
    const std::vector<ExpertWeights> weights = makeWeights(backend, dims, options.seed + 73);
    GraphCache graphs(dims);
    metal::SharedEvent event = backend.newSharedEvent();
    uint64_t eventValue = 0;
    for (uint32_t rows : rowSizes) {
      std::vector<Chunk> localCalibration{makeRoutes(rows, dims.experts, dims.topK, options.seed + rows),
                                           makeRoutes(rows, dims.experts, dims.topK, options.seed + rows + 1)};
      const Planner localPlan = Planner::calibrate(localCalibration,
                                                    Config{options.groupSize, 32, options.maxCapacity, 0.95, options.minimumGraphRows});
      const Chunk evalRoutes = makeRoutes(rows, dims.experts, dims.topK, options.seed + rows + 2);
      const InputBatch input = makeInput(rows, dims.hidden, options.seed + rows + 3);
      const LayerResult result = runLayer(backend, graphs, localPlan, evalRoutes, input, weights, dims, event, eventValue, false);
      const double total = result.packingMs + result.aneMs + result.fallbackMs + result.mergeMs;
      const double padding = result.stats.capacityRows ? 100.0 * result.stats.paddedRows / result.stats.capacityRows : 0.0;
      const double overflow = result.stats.assignments ? 100.0 * result.stats.fallbackAssignments / result.stats.assignments : 0.0;
      std::printf("layer: rows=%u E=%u topK=%u groups=%zu assignments=%llu useful=%llu fallback=%llu padding=%.1f%% overflow=%.1f%% pack=%.2f ms ANE=%.2f ms fallback(not run)=%.2f ms merge=%.2f ms total-without-fallback=%.2f ms\n",
                  rows, dims.experts, dims.topK, localPlan.groups().size(),
                  static_cast<unsigned long long>(result.stats.assignments),
                  static_cast<unsigned long long>(result.stats.aneAssignments),
                  static_cast<unsigned long long>(result.stats.fallbackAssignments), padding, overflow,
                  result.packingMs, result.aneMs, result.fallbackMs, result.mergeMs, total);
    }
    return;
  }

  const Planner planner = Planner::calibrate(calibration,
                                              Config{options.groupSize, 32, options.maxCapacity, 0.95, options.minimumGraphRows});
  for (const Chunk &routes : evaluations) {
    const InputBatch input = makeInput(routes.rows, dims.hidden, options.seed + routes.rows + 3);
    const std::vector<ExpertWeights> weights = makeWeights(backend, dims, options.seed + 73);
    GraphCache graphs(dims);
    metal::SharedEvent event = backend.newSharedEvent();
    uint64_t eventValue = 0;
    const LayerResult result = runLayer(backend, graphs, planner, routes, input, weights, dims, event, eventValue, false);
    const double total = result.packingMs + result.aneMs + result.fallbackMs + result.mergeMs;
    const double padding = result.stats.capacityRows ? 100.0 * result.stats.paddedRows / result.stats.capacityRows : 0.0;
    const double overflow = result.stats.assignments ? 100.0 * result.stats.fallbackAssignments / result.stats.assignments : 0.0;
    std::printf("layer: route-file rows=%u E=%u topK=%u groups=%zu assignments=%llu useful=%llu fallback=%llu padding=%.1f%% overflow=%.1f%% pack=%.2f ms ANE=%.2f ms fallback(not run)=%.2f ms merge=%.2f ms total-without-fallback=%.2f ms\n",
                routes.rows, dims.experts, dims.topK, planner.groups().size(),
                static_cast<unsigned long long>(result.stats.assignments),
                static_cast<unsigned long long>(result.stats.aneAssignments),
                static_cast<unsigned long long>(result.stats.fallbackAssignments), padding, overflow,
                result.packingMs, result.aneMs, result.fallbackMs, result.mergeMs, total);
  }
}


// The split's GPU side over real routes: mark packs the ANE's routes into one
// group's surfaces, the groups evaluate, join writes their results where the
// combine reads them. One command each of pack and join; the ANE runs between
// them. No GPU expert work runs beside the ANE here (the layer mode measures
// that path), so the times read as handoff plus ANE.
struct PipelineGroup final {
  uint32_t base = 0, capacity = 0, rows = 0;
  std::vector<uint32_t> experts;
  GroupWork work;
  metal::MetalBuffer scaled;  // the pack's fp16 rows before the transpose
  std::unique_ptr<Graph> graph;  // one program a group, bound to its own weights
  // One shared event a group: every evaluation's wait takes a value of its
  // own, so no other group's signal can release its join.
  metal::SharedEvent event;
  uint64_t eventValue = 0, ready = 0, done = 0;
};

void pipeline(metal::MetalBackend &backend, const Options &options) {
  if (options.routes.empty()) throw std::invalid_argument("pipeline needs --routes=FILE");
  Dimensions dims{options.hidden, options.intermediate, options.experts, options.topK};
  std::vector<Chunk> chunks = loadRoutes(options.routes, dims.experts, dims.topK);
  if (chunks.size() < 2) throw std::invalid_argument("route-file mode needs at least two chunks");
  const std::vector<Chunk> calibration(chunks.begin(), chunks.end() - 1);
  const Chunk &routes = chunks.back();
  const Planner planner = Planner::calibrate(
      calibration, Config{options.groupSize, 32, options.maxCapacity, 0.95, options.minimumGraphRows});

  const uint32_t routeCount = routes.rows * dims.topK;
  const uint32_t rowsOffset = routeCount;  // the GPU's grouped rows are below it
  std::vector<uint32_t> expertBase(dims.experts, ANE_MOE_EXPERT_ABSENT), expertCapacity(dims.experts, 0);
  std::vector<PipelineGroup> groups;
  uint32_t aneRows = 0;
  for (const Group &source : planner.groups()) {
    PipelineGroup group;
    group.base = aneRows;
    group.capacity = source.capacity;
    group.experts = source.experts;
    group.rows = group.capacity * static_cast<uint32_t>(group.experts.size());
    for (size_t slot = 0; slot < group.experts.size(); ++slot) {
      expertBase[group.experts[slot]] = group.base + static_cast<uint32_t>(slot) * group.capacity;
      expertCapacity[group.experts[slot]] = group.capacity;
    }
    aneRows += group.rows;
    groups.push_back(std::move(group));
  }

  const auto shared = [&](uint64_t bytes, const char *label) {
    metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  };
  metal::MetalBuffer selected = shared(uint64_t{routeCount} * sizeof(uint32_t), "pipeline selected");
  metal::MetalBuffer routeRows = shared(uint64_t{routeCount} * sizeof(uint32_t), "pipeline route rows");
  metal::MetalBuffer cursors = shared(uint64_t{dims.experts} * sizeof(uint32_t), "pipeline cursors");
  metal::MetalBuffer routeList = shared(uint64_t{aneRows} * sizeof(uint32_t), "pipeline route list");
  std::memset(routeList.contents(), 0xFF, uint64_t{aneRows} * sizeof(uint32_t));
  metal::MetalBuffer baseTable = shared(uint64_t{dims.experts} * sizeof(uint32_t), "pipeline expert base");
  metal::MetalBuffer capacityTable = shared(uint64_t{dims.experts} * sizeof(uint32_t), "pipeline expert capacity");
  metal::MetalBuffer chunkRows = shared(uint64_t{routes.rows} * dims.hidden * sizeof(uint16_t), "pipeline rows");
  metal::MetalBuffer expertOutput =
      shared(uint64_t{routeCount + aneRows} * dims.hidden * sizeof(uint16_t), "pipeline expert output");
  metal::MetalBuffer status = shared(sizeof(uint32_t), "pipeline status");
  metal::MetalBuffer counts = shared(uint64_t{dims.experts} * sizeof(uint32_t), "pipeline expert counts");
  metal::MetalBuffer taken = shared(sizeof(uint32_t), "pipeline taken routes");
  auto *selectedValues = static_cast<uint32_t *>(selected.contents());
  for (uint32_t route = 0; route < routeCount; ++route) selectedValues[route] = routes.routes[route].expert;
  std::memcpy(baseTable.contents(), expertBase.data(), expertBase.size() * sizeof(uint32_t));
  std::memcpy(capacityTable.contents(), expertCapacity.data(), expertCapacity.size() * sizeof(uint32_t));
  const InputBatch input = makeInput(routes.rows, dims.hidden, options.seed);
  auto *rowValues = static_cast<__bf16 *>(chunkRows.contents());
  for (uint64_t index = 0; index < uint64_t{routes.rows} * dims.hidden; ++index)
    rowValues[index] = __bf16(input.values[index]);

  const std::vector<ExpertWeights> weights = makeWeights(backend, dims, options.seed + 73);
  uint64_t packedRows = 0;
  for (PipelineGroup &group : groups) {
    PackedGroup packed;
    packed.capacity = group.capacity;
    packed.experts = group.experts;
    group.work = prepareSurfaces(backend, packed, dims.hidden);
    group.scaled = shared(uint64_t{group.rows} * dims.hidden * sizeof(uint16_t), "pipeline scaled rows");
    // A program a group: bindings of one program that take turns across groups
    // (the dense split alternates two) serve one group's weights here.
    group.graph = std::make_unique<Graph>(static_cast<uint32_t>(group.experts.size()), group.capacity, dims);
    group.event = backend.newSharedEvent();
    bindGroup(*group.graph, group.work, packed, weights);
    packedRows += group.rows;
  }
  if (packedRows != aneRows) throw std::logic_error("pipeline groups' rows differ from the plan's");

  const std::vector<uint32_t> routeExperts = [&] {
    std::vector<uint32_t> experts(routeCount);
    for (uint32_t route = 0; route < routeCount; ++route) experts[route] = routes.routes[route].expert;
    return experts;
  }();
  const uint32_t routeThreadgroups = (routeCount + 255) / 256;
  std::vector<double> totalTimes;
  uint32_t aneRoutes = 0, fallbackRoutes = 0;
  constexpr int kRepeats = 4;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    std::memcpy(selected.contents(), routeExperts.data(), routeExperts.size() * sizeof(uint32_t));
    // Clear the cursors once, and each group's scales before its pack.
    metal::CommandGraph command;
    command.add("ane_moe_clear", {cursors, groups.front().work.tx.buffer, routeList},
                AneMoeClearParams{dims.experts, 0, groups.front().base}, {1, 1, 1});
    for (const PipelineGroup &group : groups)
      command.add("ane_moe_clear", {cursors, group.work.tx.buffer, routeList},
                  AneMoeClearParams{0, group.rows, group.base}, {1, 1, 1});
    command.add("ane_moe_mark", {selected, routeRows, baseTable, capacityTable, cursors, routeList, counts, taken},
                AneMoeMarkParams{routeCount, dims.experts, rowsOffset, 0}, {routeThreadgroups, 1, 1});
    for (const PipelineGroup &group : groups) {
      command.add("ane_moe_scale", {chunkRows, routeList, group.scaled, group.work.tx.buffer},
                  AneMoeScaleParams{dims.topK, dims.hidden, group.base, group.rows}, {group.rows, 1, 1});
      command.add("ane_ffn_pack", {group.scaled, group.work.x.buffer},
                  AneFfnPackParams{dims.hidden, 0, group.work.x.strideBytes},
                  {(group.rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE, dims.hidden / ANE_FFN_TILE, 1},
                  {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
    }
    // The ANE starts once the chunk's rows are packed (the command's signal)
    // and each group's join waits inside the command for its evaluation: the
    // order AneFfn's handoff keeps, which makes the ANE's writes visible to
    // the GPU's reads.
    for (PipelineGroup &group : groups) {
      group.ready = group.eventValue += 1;
      group.done = group.eventValue += 1;
      command.signal(group.event, group.ready);
    }
    for (size_t index = 0; index < groups.size(); ++index) {
      command.wait(groups[index].event, groups[index].done);
      command.add("ane_moe_join", {groups[index].work.y.buffer, expertOutput, status},
                  AneMoeJoinParams{rowsOffset, groups[index].base, groups[index].rows, dims.hidden,
                                   groups[index].work.y.strideBytes / 2},
                  {groups[index].rows, 1, 1});
    }
    const auto start = Clock::now();
    auto ticket = backend.submitCommandAsync(command.command());
    std::vector<std::shared_ptr<GraphCompletion>> completions;
    completions.reserve(groups.size());
    for (size_t index = 0; index < groups.size(); ++index)
      completions.push_back(groups[index].graph->enqueue(*groups[index].work.binding, groups[index].event,
                                                         groups[index].ready, groups[index].done));
    static_cast<void>(ticket.wait());
    // The service's reports, as AneFfn's finish() waits for them: the next
    // application reuses the program, whose intermediate memory its
    // evaluations share.
    for (const auto &completion : completions) Graph::await(completion);
    const auto end = Clock::now();
    if (repeat) totalTimes.push_back(Seconds(end - start).count() * 1e3);
    if (!repeat) {
      const auto *marked = static_cast<const uint32_t *>(selected.contents());
      for (uint32_t route = 0; route < routeCount; ++route)
        (marked[route] == ANE_MOE_ROUTE_NONE ? aneRoutes : fallbackRoutes)++;
    }
  }

  // Every route once: the ANE's and the GPU's; the ANE's rows; and a sample
  // of the ANE's rows against the fp32 expert FFN.
  std::vector<uint32_t> claimed(dims.experts, 0);
  for (uint32_t route = 0; route < routeCount; ++route)
    if (selectedValues[route] != ANE_MOE_ROUTE_NONE) ++claimed[routes.routes[route].expert];
  uint64_t overflow = 0;
  for (uint32_t expert = 0; expert < dims.experts; ++expert)
    if (claimed[expert] > expertCapacity[expert]) overflow += claimed[expert] - expertCapacity[expert];
  const auto *rows = static_cast<const uint32_t *>(routeRows.contents());
  const auto *out = static_cast<const __bf16 *>(expertOutput.contents());
  uint32_t sampled = 0;
  // The ANE's rows against the fp32 expert FFN, 64 of them: the pipeline's
  // joins wait inside the command for their evaluations, as AneFfn's do, so
  // the ANE's outputs are what the command's joins read.
  for (uint32_t route = 0; options.verify && route < routeCount && sampled < 64; ++route) {
    if (selectedValues[route] != ANE_MOE_ROUTE_NONE) continue;
    const uint32_t row = rows[route];
    if (row < rowsOffset || row >= rowsOffset + aneRows) throw std::runtime_error("join row outside the ANE rows");
    const uint32_t column = row - rowsOffset;
    const PipelineGroup *owner = nullptr;
    for (const PipelineGroup &group : groups)
      if (column >= group.base && column < group.base + group.rows) owner = &group;
    if (!owner) throw std::runtime_error("join row outside every group");
    const auto *y = static_cast<const _Float16 *>(owner->work.y.buffer.contents());
    const uint64_t stride = owner->work.y.strideBytes / 2;
    const uint32_t local = column - owner->base;
    const uint32_t expert = routes.routes[route].expert;
    const uint32_t token = route / dims.topK;
    const std::vector<float> expected = expertForward(weights.at(expert), input, token, dims);
    double error = 0, magnitude = 0;
    for (uint32_t dim = 0; dim < dims.hidden; ++dim) {
      const __bf16 joined = out[uint64_t{row} * dims.hidden + dim];
      const __bf16 ane = __bf16(float(y[uint64_t{dim} * stride + local]));
      if (std::bit_cast<uint16_t>(joined) != std::bit_cast<uint16_t>(ane)) {
        const auto *listNow = static_cast<const uint32_t *>(routeList.contents());
        std::printf("join mismatch: route %u row %u col %u local %u dim %u joined %.6f ane %.6f expected %.6f "
                    "routeRows %u list[col] %u\n",
                    route, row, column, local, dim, double(float(joined)), double(float(ane)),
                    double(expected[dim]), rows[route], listNow[column]);
        throw std::runtime_error("join wrote another value than the ANE's output");
      }
      const double difference = double(float(joined)) - double(expected[dim]);
      error += difference * difference;
      magnitude += double(expected[dim]) * double(expected[dim]);
    }
    if (std::sqrt(error) > 0.05 * std::sqrt(magnitude) + 1e-3)
      throw std::runtime_error("ANE expert row differs from the fp32 forward by more than 5%");
    ++sampled;
  }
  if (aneRoutes + fallbackRoutes != routeCount) throw std::runtime_error("routes lost or counted twice");
  const auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  const double totalMs = median(totalTimes);
  std::printf("pipeline: rows=%u E=%u topK=%u groups=%zu ane=%u fallback=%u capacityRows=%u overflow=%llu "
              "sampled=%u total=%.3f ms\n",
              routes.rows, dims.experts, dims.topK, groups.size(), aneRoutes, fallbackRoutes, aneRows,
              static_cast<unsigned long long>(overflow), sampled, totalMs);
}

Options parseOptions(int argc, char **argv, int begin) {
  Options options;
  for (int index = begin; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const size_t equals = argument.find('=');
    if (equals == std::string_view::npos || !argument.starts_with("--"))
      throw std::invalid_argument("options use --name=value syntax");
    const std::string_view name = argument.substr(2, equals - 2);
    const std::string value(argument.substr(equals + 1));
    if (name == "experts") options.experts = std::stoul(value);
    else if (name == "top-k") options.topK = std::stoul(value);
    else if (name == "group-size") options.groupSize = std::stoul(value);
    else if (name == "max-capacity") options.maxCapacity = std::stoul(value);
    else if (name == "rows") options.rows = std::stoul(value);
    else if (name == "min-rows") options.minimumGraphRows = std::stoul(value);
    else if (name == "hidden") options.hidden = std::stoul(value);
    else if (name == "verify") options.verify = value == "on";
    else if (name == "intermediate") options.intermediate = std::stoul(value);
    else if (name == "seed") options.seed = std::stoull(value);
    else if (name == "routes") options.routes = value;
    else if (name == "seconds") options.seconds = std::stod(value);
    else throw std::invalid_argument("unknown option: " + std::string(name));
  }
  if (!options.experts || !options.topK || !options.groupSize || !options.maxCapacity ||
      options.maxCapacity % 32)
    throw std::invalid_argument("experts, top-k, group-size and a multiple-of-32 max-capacity must be positive");
  return options;
}

void usage(const char *program) {
  std::printf("Usage: %s <plan-check|check|sweep|layer|stress|pipeline> [--experts=N --top-k=N --group-size=N "
              "--max-capacity=N --min-rows=N --hidden=N --intermediate=N --rows=N --seed=N --routes=FILE "
              "--seconds=S --verify=on|off]\n"
              "stress evaluates one bound G=group-size graph of --rows rows back to back for --seconds.\n"
              "Route CSV rows: chunk,token,expert,weight (one record per assignment).\n"
              "layer defaults to Inkling dimensions and chunk sizes 512,1024,2048; "
              "--routes=FILE uses all but the last chunk for calibration.\n",
              program);
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  @autoreleasepool {
    try {
      const std::string mode = argc > 1 ? argv[1] : "plan-check";
      if (mode == "plan-check") {
        planCheck();
        return 0;
      }
      if (mode != "check" && mode != "sweep" && mode != "layer" && mode != "stress" && mode != "pipeline") {
        usage(argv[0]);
        return 2;
      }
      const Options options = parseOptions(argc, argv, 2);
      if (const auto unavailable = ane::unavailable()) {
        std::fprintf(stderr, "Neural Engine unavailable: %s\n", unavailable->c_str());
        return 2;
      }
      metal::MetalBackend backend("build/splash.metallib");
      if (mode == "check") check(backend, options);
      else if (mode == "sweep") sweep(backend);
      else if (mode == "stress") stress(backend, options);
      else if (mode == "pipeline") pipeline(backend, options);
      else layer(backend, options);
      return 0;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "ANE MoE benchmark failed: %s\n", error.what());
      return 1;
    }
  }
}