#pragma once

#include <cstdint>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <vecflow_chamfer/types.hpp>

namespace vecflow_chamfer {

// Run MaxIVF search over `queries` and write the top-`params.final_topk` doc
// IDs / scores into the caller-provided device matrices.
//
//   neighbors: [n_queries, params.final_topk] device matrix of doc IDs
//   distances: [n_queries, params.final_topk] device matrix of Chamfer scores
//
// Rows that produce fewer than `params.final_topk` valid hits are padded with
// `UINT32_MAX` doc IDs and `-FLT_MAX` scores. The doc IDs are returned in
// descending score order.
void search(raft::device_resources const& res,
            search_params const& params,
            index   const& idx,
            dataset const& ds,
            query_set const& queries,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float,    int64_t> distances);

// Same as above plus a per-stage timing breakdown / candidate-set debug dump.
// `stats.first_stage_unique_doc_ids` is populated with the post-dedup
// candidate IDs from stage 1 (one host vector per query).
void search(raft::device_resources const& res,
            search_params const& params,
            index   const& idx,
            dataset const& ds,
            query_set const& queries,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float,    int64_t> distances,
            search_stats& stats);

} // namespace vecflow_chamfer
