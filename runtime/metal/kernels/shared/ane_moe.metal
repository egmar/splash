#include "metal/abi/AneMoe.h"
#include "metal/abi/KernelABI.h"

// The GPU side of a MoE split with the Neural Engine (ops::MoE's split
// encode): a chunk's routes whose experts the ANE holds leave the GPU's
// grouping and are packed into the ANE's int8 input surfaces; the ANE's down
// outputs are copied into the rows the combine reads. The ANE multiplies int8
// weights and activations with per-row fp16 scales, without the dense split's
// Hadamard rotation: an expert's weight rows spread less than a dense FFN's and
// the rotation cost (a Metal kernel a row) earns less here.

// The cursors and scales: zero scales make a padding column's products vanish.
kernel void ane_moe_clear(device uint *cursors [[buffer(0)]], device half *scales [[buffer(1)]],
                          device uint *route_list [[buffer(2)]], constant AneMoeClearParams &p [[buffer(3)]],
                          uint thread_index [[thread_index_in_threadgroup]]) {
  for (uint index = thread_index; index < p.experts; index += 256) cursors[index] = 0;
  for (uint index = thread_index; index < p.columns; index += 256) {
    scales[index] = half(0.0f);
    route_list[p.base + index] = ANE_MOE_ROUTE_NONE;
  }
}

// A route the ANE takes leaves the GPU: its expert claims a packed row, and a
// route past the expert's capacity stays on the GPU. Runs before the grouping,
// which skips what ANE_MOE_ROUTE_NONE marks.
kernel void ane_moe_mark(device uint *selected [[buffer(0)]], device uint *route_rows [[buffer(1)]],
                         device const uint *expert_base [[buffer(2)]], device const uint *expert_capacity [[buffer(3)]],
                         device atomic_uint *cursors [[buffer(4)]], device uint *route_list [[buffer(5)]],
                         constant AneMoeMarkParams &p [[buffer(6)]], uint index [[thread_position_in_grid]]) {
  if (index >= p.routes) return;
  const uint expert = selected[index];
  if (expert >= p.experts) return;  // a shared slice, or the ANE's already
  const uint base = expert_base[expert];
  if (base == ANE_MOE_EXPERT_ABSENT) return;
  const uint slot = atomic_fetch_add_explicit(&cursors[expert], 1u, memory_order_relaxed);
  if (slot >= expert_capacity[expert]) return;  // the GPU takes this route
  route_rows[index] = p.rows_offset + base + slot;
  route_list[base + slot] = index;
  selected[index] = ANE_MOE_ROUTE_NONE;
}

// One threadgroup a column of this group: the route the group's list holds
// there, scaled to the row's peak into `scaled` (fp16, row `column`'s) with the
// scale in the group's scales; a column without a route takes zeros and scale
// 0. ane_ffn_pack transposes these rows into the ANE input surface.
kernel void ane_moe_scale(device const bfloat *input [[buffer(0)]], device const uint *route_list [[buffer(1)]],
                          device half *scaled [[buffer(2)]], device half *scales [[buffer(3)]],
                          constant AneMoeScaleParams &p [[buffer(4)]], uint column [[threadgroup_position_in_grid]],
                          uint thread_index [[thread_index_in_threadgroup]],
                          uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint simd_groups [[simdgroups_per_threadgroup]]) {
  threadgroup float peaks[32];
  if (column >= p.group_rows) return;
  device half *row = scaled + ulong(column) * p.hidden;
  const uint route = route_list[p.base + column];
  if (route == ANE_MOE_ROUTE_NONE) {
    for (uint index = thread_index; index < p.hidden; index += 256) row[index] = half(0.0f);
    if (thread_index == 0) scales[column] = half(0.0f);
    return;
  }
  const uint token = route / p.routes_per_row;
  const device bfloat *values = input + ulong(token) * p.hidden;
  float peak = 0.0f;
  for (uint index = thread_index; index < p.hidden; index += 256) peak = max(peak, fabs(float(values[index])));
  peak = simd_max(peak);
  if (simd_lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak, 0x1p-12f) / 127.0f, inverse = 1.0f / scale;
  for (uint index = thread_index; index < p.hidden; index += 256)
    row[index] = half(float(values[index]) * inverse);
  if (thread_index == 0) scales[column] = half(scale);
}

// One copy a group: its ANE output rows into the routes' expert-output rows,
// where the combine multiplies them by the routes' weights. A route of another
// group, or none, writes nothing.
kernel void ane_moe_join(device const half *ane_output [[buffer(0)]], device const uint *route_list [[buffer(1)]],
                         device bfloat *expert_output [[buffer(2)]], constant AneMoeJoinParams &p [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint thread_index [[thread_index_in_threadgroup]]) {
  const uint column = group.x;
  if (column >= p.group_rows) return;
  const uint route = route_list[p.base + column];
  if (route == ANE_MOE_ROUTE_NONE) return;
  const uint row = p.rows_offset + p.base + column;
  for (uint dim = thread_index; dim < p.hidden; dim += 256)
    expert_output[ulong(row) * p.hidden + dim] = bfloat(float(ane_output[ulong(dim) * p.stride + column]));
}
