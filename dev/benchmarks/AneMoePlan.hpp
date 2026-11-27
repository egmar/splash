#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::dev::ane_moe {

struct Route final {
  uint32_t token = 0;
  uint32_t expert = 0;
  float weight = 0.0f;
};

struct Chunk final {
  uint32_t rows = 0;
  uint32_t experts = 0;
  uint32_t topK = 0;
  std::vector<Route> routes;

  void validate() const {
    if (!rows || !experts || !topK || topK > experts || routes.size() != uint64_t{rows} * topK)
      throw std::invalid_argument("invalid route chunk dimensions");
    std::vector<uint32_t> perToken(rows);
    std::vector<uint8_t> seen(uint64_t{rows} * experts);
    for (const Route &route : routes) {
      if (route.token >= rows || route.expert >= experts || !std::isfinite(route.weight) || route.weight < 0.0f)
        throw std::invalid_argument("route is outside its chunk or has an invalid weight");
      uint8_t &duplicate = seen[uint64_t{route.token} * experts + route.expert];
      if (duplicate) throw std::invalid_argument("a token routes to one expert more than once");
      duplicate = 1;
      ++perToken[route.token];
      if (perToken[route.token] > topK) throw std::invalid_argument("too many routes for a token");
    }
    for (uint32_t count : perToken)
      if (count != topK) throw std::invalid_argument("route chunk does not have a fixed top-k per token");
  }
};

struct Config final {
  uint32_t groupSize = 4;
  uint32_t capacityQuantum = 32;
  uint32_t maxCapacity = 256;
  double loadQuantile = 0.95;
  uint32_t minimumGraphRows = 0;
};

struct ExpertPlan final {
  uint32_t capacity = 0;
  uint32_t group = UINT32_MAX;
  uint32_t groupSlot = UINT32_MAX;
};

struct Group final {
  uint32_t capacity = 0;
  std::vector<uint32_t> experts;
};

struct PackedRoute final {
  uint32_t routeIndex = UINT32_MAX;
  uint32_t token = 0;
  uint32_t expert = 0;
  uint32_t group = UINT32_MAX;
  uint32_t row = UINT32_MAX;
};

struct PackedGroup final {
  uint32_t capacity = 0;
  std::vector<uint32_t> experts;
  // Expert-major rows; unused capacity slots have routeIndex == UINT32_MAX.
  std::vector<PackedRoute> rows;
};

struct PackStats final {
  uint64_t assignments = 0;
  uint64_t aneAssignments = 0;
  uint64_t fallbackAssignments = 0;
  uint64_t paddedRows = 0;
  uint64_t capacityRows = 0;
};

struct Packed final {
  std::vector<PackedGroup> groups;
  std::vector<uint32_t> fallbackRoutes;
  PackStats stats;

  [[nodiscard]] bool coversExactly(const Chunk &chunk) const {
    if (stats.assignments != chunk.routes.size() ||
        stats.aneAssignments + stats.fallbackAssignments != stats.assignments ||
        fallbackRoutes.size() != stats.fallbackAssignments)
      return false;
    std::vector<uint8_t> seen(chunk.routes.size());
    for (uint32_t route : fallbackRoutes) {
      if (route >= seen.size() || seen[route]++) return false;
    }
    for (const PackedGroup &group : groups) {
      if (group.rows.size() != uint64_t{group.experts.size()} * group.capacity) return false;
      for (const PackedRoute &route : group.rows) {
        if (route.routeIndex == UINT32_MAX) continue;
        if (route.routeIndex >= seen.size() || seen[route.routeIndex]++) return false;
        const Route &source = chunk.routes[route.routeIndex];
        if (source.token != route.token || source.expert != route.expert) return false;
      }
    }
    return std::ranges::all_of(seen, [](uint8_t count) { return count == 1; });
  }
};

class Planner final {
public:
  [[nodiscard]] static Planner calibrate(std::span<const Chunk> chunks, Config config = {}) {
    if (chunks.empty()) throw std::invalid_argument("calibration needs at least one route chunk");
    if (!config.groupSize || !config.capacityQuantum || !config.maxCapacity ||
        config.maxCapacity < config.capacityQuantum || config.maxCapacity % config.capacityQuantum ||
        !(config.loadQuantile > 0.0 && config.loadQuantile <= 1.0))
      throw std::invalid_argument("invalid ANE MoE planner configuration");

    const uint32_t expertCount = chunks.front().experts;
    std::vector<std::vector<uint32_t>> counts(expertCount, std::vector<uint32_t>(chunks.size()));
    for (size_t batch = 0; batch < chunks.size(); ++batch) {
      chunks[batch].validate();
      if (chunks[batch].experts != expertCount) throw std::invalid_argument("calibration expert counts differ");
      for (const Route &route : chunks[batch].routes) ++counts[route.expert][batch];
    }

    Planner planner;
    planner.config_ = config;
    planner.experts_.resize(expertCount);
    std::map<uint32_t, std::vector<uint32_t>> byCapacity;
    for (uint32_t expert = 0; expert < expertCount; ++expert) {
      auto &loads = counts[expert];
      std::sort(loads.begin(), loads.end());
      const size_t rank = static_cast<size_t>(std::ceil(config.loadQuantile * loads.size())) - 1;
      const uint32_t quantile = loads[rank];
      if (!quantile) continue; // Cold at calibration: keep exact routes on the fallback.
      const uint64_t rounded = ((uint64_t{quantile} + config.capacityQuantum - 1) / config.capacityQuantum) *
                               config.capacityQuantum;
      const uint32_t capacity = static_cast<uint32_t>(std::min<uint64_t>(rounded, config.maxCapacity));
      planner.experts_[expert].capacity = capacity;
      byCapacity[capacity].push_back(expert);
    }

    for (const auto &[capacity, ids] : byCapacity) {
      for (size_t begin = 0; begin < ids.size(); begin += config.groupSize) {
        Group group;
        const size_t end = std::min(ids.size(), begin + config.groupSize);
        group.experts.assign(ids.begin() + begin, ids.begin() + end);
        const uint64_t minimumCapacity = config.minimumGraphRows
                                             ? ((uint64_t{config.minimumGraphRows} + group.experts.size() - 1) /
                                                group.experts.size() + config.capacityQuantum - 1) /
                                                   config.capacityQuantum * config.capacityQuantum
                                             : 0;
        group.capacity = static_cast<uint32_t>(std::max<uint64_t>(capacity, minimumCapacity));
        if (group.capacity > config.maxCapacity)
          throw std::invalid_argument("max capacity cannot satisfy the ANE minimum graph row count");
        const uint32_t groupIndex = static_cast<uint32_t>(planner.groups_.size());
        for (uint32_t slot = 0; slot < group.experts.size(); ++slot) {
          ExpertPlan &expert = planner.experts_[group.experts[slot]];
          expert.capacity = group.capacity;
          expert.group = groupIndex;
          expert.groupSlot = slot;
        }
        planner.groups_.push_back(std::move(group));
      }
    }
    return planner;
  }

  [[nodiscard]] Packed pack(const Chunk &chunk) const {
    chunk.validate();
    if (chunk.experts != experts_.size()) throw std::invalid_argument("route chunk expert count differs from plan");

    Packed packed;
    packed.stats.assignments = chunk.routes.size();
    packed.groups.reserve(groups_.size());
    std::vector<std::vector<uint32_t>> perExpert(experts_.size());
    for (uint32_t route = 0; route < chunk.routes.size(); ++route)
      perExpert[chunk.routes[route].expert].push_back(route);

    for (uint32_t groupIndex = 0; groupIndex < groups_.size(); ++groupIndex) {
      const Group &sourceGroup = groups_[groupIndex];
      PackedGroup group;
      group.capacity = sourceGroup.capacity;
      group.experts = sourceGroup.experts;
      group.rows.resize(uint64_t{group.capacity} * group.experts.size());
      packed.stats.capacityRows += group.rows.size();
      for (uint32_t slot = 0; slot < group.experts.size(); ++slot) {
        const uint32_t expert = group.experts[slot];
        auto &routes = perExpert[expert];
        const uint32_t accepted = std::min<uint32_t>(routes.size(), group.capacity);
        for (uint32_t row = 0; row < accepted; ++row) {
          const uint32_t routeIndex = routes[row];
          const Route &route = chunk.routes[routeIndex];
          group.rows[uint64_t{slot} * group.capacity + row] =
              {routeIndex, route.token, route.expert, groupIndex, row};
        }
        packed.stats.aneAssignments += accepted;
        for (uint32_t route = accepted; route < routes.size(); ++route) packed.fallbackRoutes.push_back(routes[route]);
      }
      packed.groups.push_back(std::move(group));
    }
    for (uint32_t expert = 0; expert < experts_.size(); ++expert)
      if (experts_[expert].capacity == 0)
        packed.fallbackRoutes.insert(packed.fallbackRoutes.end(), perExpert[expert].begin(), perExpert[expert].end());
    packed.stats.fallbackAssignments = packed.fallbackRoutes.size();
    packed.stats.paddedRows = packed.stats.capacityRows - packed.stats.aneAssignments;
    if (!packed.coversExactly(chunk)) throw std::logic_error("planner did not preserve every route exactly once");
    return packed;
  }

  [[nodiscard]] const std::vector<ExpertPlan> &experts() const noexcept { return experts_; }
  [[nodiscard]] const std::vector<Group> &groups() const noexcept { return groups_; }
  [[nodiscard]] const Config &config() const noexcept { return config_; }

private:
  Config config_;
  std::vector<ExpertPlan> experts_;
  std::vector<Group> groups_;
};

// CSV records are `chunk,token,expert,weight`; blank lines and `#` comments
// are ignored. Every chunk must contain exactly topK distinct experts per row.
[[nodiscard]] inline std::vector<Chunk> loadRoutes(const std::filesystem::path &path, uint32_t experts,
                                                   uint32_t topK) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open route file: " + path.string());
  std::map<uint32_t, std::map<uint32_t, std::vector<Route>>> records;
  std::string line;
  size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    if (line.empty() || line.front() == '#') continue;
    uint32_t chunk = 0, token = 0, expert = 0;
    float weight = 0.0f;
    char comma1 = 0, comma2 = 0, comma3 = 0;
    std::istringstream row(line);
    if (!(row >> chunk >> comma1 >> token >> comma2 >> expert >> comma3 >> weight) || comma1 != ',' ||
        comma2 != ',' || comma3 != ',')
      throw std::runtime_error("invalid route record at line " + std::to_string(lineNumber));
    records[chunk][token].push_back({token, expert, weight});
  }
  if (records.empty()) throw std::runtime_error("route file contains no records");
  std::vector<Chunk> chunks;
  for (const auto &[chunkId, rows] : records) {
    static_cast<void>(chunkId);
    Chunk chunk;
    chunk.rows = rows.rbegin()->first + 1;
    chunk.experts = experts;
    chunk.topK = topK;
    chunk.routes.reserve(uint64_t{chunk.rows} * topK);
    for (uint32_t token = 0; token < chunk.rows; ++token) {
      const auto found = rows.find(token);
      if (found == rows.end()) throw std::runtime_error("route file has a missing token row");
      chunk.routes.insert(chunk.routes.end(), found->second.begin(), found->second.end());
    }
    chunk.validate();
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

} // namespace splash::dev::ane_moe