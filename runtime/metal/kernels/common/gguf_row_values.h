#pragma once

#include "metal/kernels/common/quant_formats.h"

// A GGUF image tensor's values, four at a time, as the ANE splits' weight
// kernels read them (kernels/prefill/ane_ffn.metal, kernels/shared/ane_moe.metal):
// an image tensor of format F with `groups` groups of 32 inputs a row has its
// row `row`'s values [input, input + 4) at pairs 2h and 2h + 1 of chunk c of
// their group, h = input bit 4 and c = input bits 2, 3 (metal/abi/QuantFormat.h),
// as kernels/common/gguf_staged.h decodes them. Only that chunk is read (and
// chunk 0, which holds the scale of a format whose scale is in every chunk).
template <class F>
inline void gguf_row_values(thread float (&value)[4], device uchar *plane0, device uchar *plane1, device uchar *meta,
                            uint groups, uint row, uint input) {
  const uint group = input >> 5, h = (input >> 4) & 1;
  const ushort c = (input >> 2) & 3;
  const ulong unit = quant_tile_index(row, group, groups);
  device uchar *p0 = plane0 + unit * F::P0, *p1 = plane1 + unit * F::P1;
  const typename F::Meta header =
      F::loadMeta(meta + quant_tile_index(row, group / F::MetaGroups, groups / F::MetaGroups) * F::MetaBytes);
  const typename F::Chunk q = F::loadChunk(p0, p1, c);
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(header, F::loadChunk(p0, p1, 0));
  else k = F::coef(header, ushort(group % F::MetaGroups));
  const float s = h ? k.s.y : k.s.x, m = h ? k.m.y : k.m.x;
  float4 v;
  if constexpr (F::Kind == QuantLinear) {
    const uint4 pairs = F::codes(q);
    const uint a = h ? pairs.z : pairs.x, b = h ? pairs.w : pairs.y;
    const float4 code = float4(a & 0xFFFFu, a >> 16, b & 0xFFFFu, b >> 16) - float(F::Zero);
    v = F::Zero ? code * s : fma(code, float4(s), float4(m));
  } else if constexpr (F::Kind == QuantCodebook) {
    const uchar4 bytes = as_type<uchar4>(F::indices(q));
    const uint a = h ? bytes.z : bytes.x, b = h ? bytes.w : bytes.y;
    v = float4(F::value(a & 15), F::value(a >> 4), F::value(b & 15), F::value(b >> 4)) * s;
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 codes = F::values(q);
    v = float4(as_type<char4>(h ? codes.y : codes.x)) * s;
  } else if constexpr (F::Kind == QuantFloat8) {
    const uint2 codes = F::values(q);
    v = float4(quant_e4m3_word(h ? codes.y : codes.x)) * s;
  } else if constexpr (F::Kind == QuantFloat16) {
    typedef vec<typename F::Value, 2> V2;
    const uint4 values = F::values(q);
    v = float4(float2(as_type<V2>(h ? values.z : values.x)), float2(as_type<V2>(h ? values.w : values.y))) * s;
  } else {
    const uint2 grid = F::grid(q);
    const uint signs = F::signs(q) >> (4 * h);
    v = float4(as_type<uchar4>(h ? grid.y : grid.x)) * s;
    v = select(v, -v, bool4(signs & 1, signs & 2, signs & 4, signs & 8));
  }
  for (uint e = 0; e < 4; ++e) value[e] = v[e];
}
