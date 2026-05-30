#include "recompute_anchors.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>

namespace vecflow_chamfer::detail {

// Recompute anchor positions as the per-anchor mean of all assigned data points.
template <typename DataT>
void recompute_anchors(raft::resources const& handle,
                       const DataT* X_device,
                       raft::device_vector_view<uint32_t, int64_t> anchor_labels,
                       raft::device_matrix_view<DataT, int64_t> anchors,
                       int64_t n_samples,
                       int64_t n_features,
                       int64_t n_anchors) {

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  const int64_t anchor_batch_size = 1000000;
  const int64_t data_batch_size   = 5000000;

  // Pull labels to host once so the inner loop's host-side anchor-range
  // partitioning can be done without a roundtrip per data batch.
  std::vector<uint32_t> h_anchor_labels(n_samples);
  raft::copy(h_anchor_labels.data(), anchor_labels.data_handle(), n_samples, stream);
  raft::resource::sync_stream(handle);

  for (int64_t anchor_start = 0; anchor_start < n_anchors; anchor_start += anchor_batch_size) {
    int64_t current_anchor_batch = std::min(anchor_batch_size, n_anchors - anchor_start);
    int64_t anchor_end = anchor_start + current_anchor_batch;

    auto d_anchor_batch_sums   = raft::make_device_matrix<float, int64_t>(handle, current_anchor_batch, n_features);
    auto d_anchor_batch_counts = raft::make_device_vector<int,   int64_t>(handle, current_anchor_batch);

    raft::linalg::map_offset(handle, d_anchor_batch_sums.view(),   [] __device__(int64_t) { return 0.0f; });
    raft::linalg::map_offset(handle, d_anchor_batch_counts.view(), [] __device__(int64_t) { return 0;    });

    auto d_data_batch_X      = raft::make_device_matrix<DataT,    int64_t>(handle, data_batch_size, n_features);
    auto d_data_batch_labels = raft::make_device_vector<uint32_t, int64_t>(handle, data_batch_size);

    for (int64_t data_start = 0; data_start < n_samples; data_start += data_batch_size) {
      int64_t current_data_batch = std::min(data_batch_size, n_samples - data_start);

      raft::copy(d_data_batch_X.data_handle(),
                 X_device + data_start * n_features,
                 current_data_batch * n_features, stream);
      raft::copy(d_data_batch_labels.data_handle(),
                 h_anchor_labels.data() + data_start,
                 current_data_batch, stream);

      auto data_labels_ptr   = d_data_batch_labels.data_handle();
      auto data_X_ptr        = d_data_batch_X.data_handle();
      auto anchor_sums_ptr   = d_anchor_batch_sums.data_handle();
      auto anchor_counts_ptr = d_anchor_batch_counts.data_handle();

      thrust::for_each(raft::resource::get_thrust_policy(handle),
        thrust::counting_iterator<int64_t>(0),
        thrust::counting_iterator<int64_t>(current_data_batch),
        [=] __device__(int64_t sample_idx) {
          uint32_t anchor_id = data_labels_ptr[sample_idx];
          if (anchor_id >= anchor_start && anchor_id < anchor_end) {
            uint32_t local_anchor_id = anchor_id - anchor_start;
            atomicAdd(&anchor_counts_ptr[local_anchor_id], 1);
            for (int64_t dim = 0; dim < n_features; dim++) {
              DataT val = data_X_ptr[sample_idx * n_features + dim];
              float float_val;
              if constexpr (std::is_same_v<DataT, __half>) {
                float_val = __half2float(val);
              } else {
                float_val = static_cast<float>(val);
              }
              atomicAdd(&anchor_sums_ptr[local_anchor_id * n_features + dim], float_val);
            }
          }
        });
    }

    auto anchor_counts_float = raft::make_device_vector<float, int64_t>(handle, current_anchor_batch);
    raft::linalg::unaryOp(anchor_counts_float.data_handle(),
                          d_anchor_batch_counts.data_handle(),
                          current_anchor_batch,
                          [] __device__(int val) { return static_cast<float>(val); },
                          stream);

    // Divide each centroid row by its cluster count. raft::matrixVectorOp
    // is (out, matrix, vec, D=cols, N=rows, op, stream) with rowMajor and
    // bcastAlongRows as template non-type params; bcastAlongRows=false applies
    // one scalar per row. div_checkzero_op leaves the row at 0 when count==0.
    raft::linalg::matrixVectorOp<true /*rowMajor*/, false /*bcastAlongRows*/>(
      d_anchor_batch_sums.data_handle(),
      d_anchor_batch_sums.data_handle(),
      anchor_counts_float.data_handle(),
      static_cast<int64_t>(n_features),            // D = cols
      static_cast<int64_t>(current_anchor_batch),  // N = rows
      raft::div_checkzero_op{},
      stream);

    auto anchors_batch_view = raft::make_device_matrix_view(
      anchors.data_handle() + anchor_start * n_features,
      current_anchor_batch, n_features);

    raft::linalg::unaryOp(anchors_batch_view.data_handle(),
                          d_anchor_batch_sums.data_handle(),
                          current_anchor_batch * n_features,
                          [] __device__(float val) {
                            if constexpr (std::is_same_v<DataT, __half>) {
                              return __float2half(val);
                            } else {
                              return static_cast<DataT>(val);
                            }
                          }, stream);
  }
}

template void recompute_anchors<__half>(
  raft::resources const&,
  const __half*,
  raft::device_vector_view<uint32_t, int64_t>,
  raft::device_matrix_view<__half, int64_t>,
  int64_t, int64_t, int64_t);

} // namespace vecflow_chamfer::detail
