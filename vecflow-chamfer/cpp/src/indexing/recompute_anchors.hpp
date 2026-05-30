// Internal: anchor-mean recomputation used during MaxIVF refinement iterations.
// Not part of the public API.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

namespace vecflow_chamfer::detail {

// Recompute anchor positions as the per-anchor mean of all assigned data points.
// Body is in src/indexing/recompute_anchors.cu; only __half is explicitly
// instantiated for libvecflow_chamfer.
template <typename DataT>
void recompute_anchors(raft::resources const& handle,
                       const DataT* X_device,
                       raft::device_vector_view<uint32_t, int64_t> anchor_labels,
                       raft::device_matrix_view<DataT, int64_t> anchors,
                       int64_t n_samples,
                       int64_t n_features,
                       int64_t n_anchors);

extern template void recompute_anchors<__half>(
  raft::resources const&,
  const __half*,
  raft::device_vector_view<uint32_t, int64_t>,
  raft::device_matrix_view<__half, int64_t>,
  int64_t, int64_t, int64_t);

} // namespace vecflow_chamfer::detail
