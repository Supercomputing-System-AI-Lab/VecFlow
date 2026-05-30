// Internal: anchor -> doc_id inverted-list builder (build-time only).
#pragma once

#include <cstdint>
#include <vector>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

namespace vecflow_chamfer::detail {

void anchor_to_doc_id_mapping(raft::resources const& handle,
                              raft::device_vector_view<const uint32_t, int64_t> anchor_labels_view,
                              raft::device_vector_view<const uint32_t, int64_t> doc_ids_view,
                              int64_t n_vectors,
                              int64_t n_anchors,
                              std::vector<uint32_t>& unique_doc_ids,
                              std::vector<uint32_t>& doc_offsets);

} // namespace vecflow_chamfer::detail
