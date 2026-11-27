#pragma once

// The GPU side of a MoE split with the Neural Engine (kernels/shared/ane_moe.metal):
// a chunk's routes whose experts the ANE holds are packed into the ANE's int8
// input surfaces and taken out of the GPU's grouping; the ANE's down outputs
// are copied into the expert outputs the combine reads.
//
// `selected` holds one expert per route (metal/abi/MoE.h). A route the ANE
// takes carries ANE_MOE_ROUTE_NONE there, which the grouping skips, and
// `route_rows[route]` names the row of the combine's expert outputs its ANE
// result is written to: `rows_offset + base + slot`, where `base` is the
// expert's first row among the ANE's (the plan's table) and `slot` its packed
// row. The GPU's own grouped rows are below `rows_offset` (a chunk's routes
// bound them), so the combine tells them apart by the row alone.
#define ANE_MOE_ROUTE_NONE 0xFFFFFFFFu
// An expert the ANE does not hold, in the plan's base table.
#define ANE_MOE_EXPERT_ABSENT 0xFFFFFFFFu

// A group's columns scaled for the pack: each column's route row, scaled to
// the column's peak in `scaled` (fp16 rows of `hidden` values) with the scale
// in the group's scales. A column no route holds takes zeros and scale 0, so
// its products vanish whatever its codes hold; ane_ffn_pack (AneFfn.h) then
// transposes the scaled rows into the int8 surface's columns.
struct AneMoeScaleParams {
  uint32_t routes_per_row;
  uint32_t hidden;
  uint32_t base;         // this group's first row among the ANE's: its columns'
                         // route list begins there
  uint32_t group_rows;   // this group's rows (capacity * experts)
};

// One copy a group: its ANE outputs into the routes' expert-output rows.
struct AneMoeJoinParams {
  uint32_t rows_offset;
  uint32_t base;         // this group's first row among the ANE's
  uint32_t group_rows;   // this group's rows (capacity * experts)
  uint32_t hidden;
  uint32_t stride;       // the ANE output surface's row stride, in fp16 values
};

// The routes of a group's experts whose routes fit their experts' capacities
// leave the GPU: `selected[route]` becomes ANE_MOE_ROUTE_NONE and
// `route_rows[route]` the row the ANE's result joins at.
struct AneMoeMarkParams {
  uint32_t routes;
  uint32_t experts;
  uint32_t rows_offset;
};

// A group's cursors, scales and route list: the per-expert packed-row counters
// `cursors`, the `columns` scales of the group's input surface, and the
// `columns` route-list entries at `base` (ANE_MOE_ROUTE_NONE, no route).
struct AneMoeClearParams {
  uint32_t experts;
  uint32_t columns;
  uint32_t base;
};
