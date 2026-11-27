#include "metal/abi/AneMoe.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_row_values.h"

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
                         device atomic_uint *counts [[buffer(6)]], constant AneMoeMarkParams &p [[buffer(7)]],
                         uint index [[thread_position_in_grid]]) {
  if (index >= p.routes) return;
  const uint expert = selected[index];
  if (expert >= p.experts) return;  // a shared slice, or the ANE's already
  // What the routing chose, before the engine takes its share: a start reads
  // it to plan which experts to hold.
  atomic_fetch_add_explicit(&counts[p.block * p.experts + expert], 1u, memory_order_relaxed);
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
// where the combine multiplies them by the routes' weights. Every column is
// copied, padding's included: a padding column's scales are zero, so its rows
// are zero, and no route's route_rows names them. A value that is not finite
// sets `status`, which the host reads once the command completes.
kernel void ane_moe_join(device const half *ane_output [[buffer(0)]], device bfloat *expert_output [[buffer(1)]],
                         device atomic_uint *status [[buffer(2)]], constant AneMoeJoinParams &p [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint thread_index [[thread_index_in_threadgroup]]) {
  const uint column = group.x;
  if (column >= p.group_rows) return;
  const uint row = p.rows_offset + p.base + column;
  for (uint dim = thread_index; dim < p.hidden; dim += 256) {
    const half value = ane_output[ulong(dim) * p.stride + column];
    if ((as_type<ushort>(value) & 0x7c00) == 0x7c00) atomic_store_explicit(status, 1u, memory_order_relaxed);
    expert_output[ulong(row) * p.hidden + dim] = bfloat(float(value));
  }
}

// A block's resident experts' weights as the group's int8 rows of `scale`
// halves a row, from their GGUF planes: one simdgroup a row, ANE_FFN_WEIGHT_ROWS
// rows a threadgroup, no rotation (the split's graph takes plain int8). A row's
// codes are its values over the row's peak / 127, as ane_ffn_weights stores a
// dense FFN's, and its scale the peak / 127 * ANE_FFN_INT8_UNIT times
// `multiplier` (ops::AneMoe's headroom and the expert's scale, which the
// graph's product carries) — never the codes, which keep their precision.
template <class F>
inline void ane_moe_weight_rows(device uchar *a, device uchar *b, device uchar *c, device char *output,
                                device half *scale, constant AneMoeWeightParams &params, uint tile,
                                uint simd_group, uint lane) {
  const uint row = tile * ANE_FFN_WEIGHT_ROWS + simd_group;
  if (row >= params.rows) return;
  const uint source = params.row + row, end = params.input + params.width;
  float peak = 0.0f;
  for (uint input = params.input + lane * 4; input < end; input += 32 * 4) {
    float value[4];
    gguf_row_values<F>(value, a, b, c, params.groups, source, input);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[e]));
  }
  peak = simd_max(peak);
  const float quant = max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK, inverse = 1.0f / quant;
  for (uint input = params.input + lane * 4; input < end; input += 32 * 4) {
    float value[4];
    gguf_row_values<F>(value, a, b, c, params.groups, source, input);
    *(device char4 *)(output + ulong(row) * params.stride + (input - params.input)) =
        char4(clamp(rint(float4(value[0], value[1], value[2], value[3]) * inverse), -ANE_FFN_INT8_PEAK,
                    ANE_FFN_INT8_PEAK));
  }
  if (lane == 0) scale[row * params.scale_stride] = half(float(half(quant * ANE_FFN_INT8_UNIT)) * params.multiplier);
}

// The same with a scale a group of 32 inputs, as a dense split's rotation
// earns for an expert: each lane a group of the row, its own peak, its scale
// and its codes, so a group's outliers cost only their group's precision.
template <class F>
inline void ane_moe_weight_group_rows(device uchar *a, device uchar *b, device uchar *c, device char *output,
                                      device half *scale, constant AneMoeWeightParams &params, uint tile,
                                      uint simd_group, uint lane) {
  const uint row = tile * ANE_FFN_WEIGHT_ROWS + simd_group;
  if (row >= params.rows) return;
  const uint source = params.row + row, groups = params.width / 32;
  for (uint group = lane; group < groups; group += 32) {
    const uint begin = params.input + group * 32;
    float value[32], peak = 0.0f;
    for (uint at = 0; at < 32; at += 4) {
      float four[4];
      gguf_row_values<F>(four, a, b, c, params.groups, source, begin + at);
      for (uint e = 0; e < 4; ++e) {
        value[at + e] = four[e];
        peak = max(peak, fabs(four[e]));
      }
    }
    const float quant = max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK, inverse = 1.0f / quant;
    for (uint at = 0; at < 32; at += 4)
      *(device char4 *)(output + ulong(row) * params.stride + group * 32 + at) =
          char4(clamp(rint(float4(value[at], value[at + 1], value[at + 2], value[at + 3]) * inverse),
                      -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
    scale[ulong(row) * params.scale_stride + group] =
        half(float(half(quant * ANE_FFN_INT8_UNIT)) * params.multiplier);
  }
}

// A projection of a resident expert (ops::AneMoe), a scale a row.
kernel void ane_moe_weights_grouped_gguf(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]],
                                        device uchar *c [[buffer(2)]], device char *output [[buffer(3)]],
                                        device half *scale [[buffer(4)]],
                                        constant AneMoeWeightParams &params [[buffer(5)]],
                                        uint tile [[threadgroup_position_in_grid]],
                                        uint simd_group [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
  quant_format_switch(params.format, [&](auto format) {
    ane_moe_weight_group_rows<decltype(format)>(a, b, c, output, scale, params, tile, simd_group, lane);
  });
}

// A projection of a resident expert (ops::AneMoe), the body of the dispatch's
// format.
kernel void ane_moe_weights_gguf(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]],
                                 device uchar *c [[buffer(2)]], device char *output [[buffer(3)]],
                                 device half *scale [[buffer(4)]], constant AneMoeWeightParams &params [[buffer(5)]],
                                 uint tile [[threadgroup_position_in_grid]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  quant_format_switch(params.format, [&](auto format) {
    ane_moe_weight_rows<decltype(format)>(a, b, c, output, scale, params, tile, simd_group, lane);
  });
}
