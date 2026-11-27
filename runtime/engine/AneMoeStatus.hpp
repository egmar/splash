#pragma once

#include <cstdint>
#include <string>

namespace splash::engine {

// The prefill MoE's Neural Engine split as /status reports it (ops::AneMoe):
// whether it runs, the experts each block keeps and their packed rows, the
// memory it holds, and what it ran.
struct AneMoeSnapshot {
  enum class State : uint8_t { Off, Split, Stopped };
  State state = State::Off;
  // Why it stopped serving, or why the start took none.
  std::string reason;
  // Each block's copies the engine holds, the rows each holds, and the blocks
  // it holds them of (ops::AneMoe::Plan).
  uint32_t perBlock = 0, rowsPerCopy = 0, blocks = 0;
  // The rows each step's expert outputs hold for its join
  // (ops::AneMoe::regionRows).
  uint32_t joinRows = 0;
  // What it allocated, and what it ran (ops::AneMoe::Served).
  uint64_t allocatedBytes = 0;
  uint64_t commands = 0, evaluations = 0;
};

} // namespace splash::engine
