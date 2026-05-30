// Internal: VPQ side-encode helper. Pulled into its own TU so the cuVS PQ
// header (and its transitive cluster/kmeans includes) never bleed into the
// rest of the build pipeline.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <raft/core/device_resources.hpp>

#include <vecflow_chamfer/types.hpp>

namespace vecflow_chamfer::detail {

// Train a PQ codebook on per-token residuals (token - anchor_centroid) and
// encode every token. The MaxIVF anchors play the VQ role, so no separate
// VQ codebook is trained.
//
//   d_anchor_labels    : [n_doc_tokens] uint32 device — per-token VQ index
//   d_anchor_centroids : [n_anchors, dim] fp16 device — VQ centroids (cagra dataset)
vpq_data build_vpq(raft::device_resources const& res,
                   dataset const& ds,
                   uint32_t* d_anchor_labels,
                   const __half* d_anchor_centroids,
                   uint32_t pq_bits,
                   uint32_t pq_dim);

} // namespace vecflow_chamfer::detail
