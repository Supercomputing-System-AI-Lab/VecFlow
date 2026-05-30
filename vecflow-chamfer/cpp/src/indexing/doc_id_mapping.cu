#include "doc_id_mapping.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <raft/core/copy.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>

namespace {

struct make_pair_functor {
  uint32_t* anchor_labels;
  uint32_t* doc_ids;

  make_pair_functor(uint32_t* a_labels, uint32_t* d_ids)
    : anchor_labels(a_labels), doc_ids(d_ids) {}

  __device__
  thrust::tuple<uint32_t, uint32_t> operator()(int64_t i) const {
    return thrust::make_tuple(anchor_labels[i], doc_ids[i]);
  }
};

struct get_anchor_functor {
  __device__
  uint32_t operator()(const thrust::tuple<uint32_t, uint32_t>& pair) const {
    return thrust::get<0>(pair);
  }
};

struct get_doc_functor {
  __device__
  uint32_t operator()(const thrust::tuple<uint32_t, uint32_t>& pair) const {
    return thrust::get<1>(pair);
  }
};

}  // namespace

namespace vecflow_chamfer::detail {

void anchor_to_doc_id_mapping(raft::resources const& handle,
                              raft::device_vector_view<const uint32_t, int64_t> anchor_labels_view,
                              raft::device_vector_view<const uint32_t, int64_t> doc_ids_view,
                              int64_t n_vectors,
                              int64_t n_anchors,
                              std::vector<uint32_t>& unique_doc_ids,
                              std::vector<uint32_t>& doc_offsets) {

  uint32_t* anchor_labels = const_cast<uint32_t*>(anchor_labels_view.data_handle());
  uint32_t* doc_ids = const_cast<uint32_t*>(doc_ids_view.data_handle());

  auto thrust_policy = raft::resource::get_thrust_policy(handle);

  thrust::device_vector<thrust::tuple<uint32_t, uint32_t>> anchor_doc_pairs(n_vectors);
  thrust::transform(thrust_policy,
    thrust::make_counting_iterator<int64_t>(0),
    thrust::make_counting_iterator<int64_t>(n_vectors),
    anchor_doc_pairs.begin(),
    make_pair_functor(anchor_labels, doc_ids));

  thrust::sort(thrust_policy, anchor_doc_pairs.begin(), anchor_doc_pairs.end());
  auto new_end = thrust::unique(thrust_policy, anchor_doc_pairs.begin(), anchor_doc_pairs.end());
  int64_t unique_count = new_end - anchor_doc_pairs.begin();
  anchor_doc_pairs.resize(unique_count);

  thrust::device_vector<uint32_t> d_unique_doc_ids(unique_count);
  thrust::device_vector<uint32_t> anchor_ids(unique_count);

  thrust::transform(thrust_policy,
    anchor_doc_pairs.begin(), anchor_doc_pairs.end(),
    anchor_ids.begin(),
    get_anchor_functor());

  thrust::transform(thrust_policy,
    anchor_doc_pairs.begin(), anchor_doc_pairs.end(),
    d_unique_doc_ids.begin(),
    get_doc_functor());

  thrust::device_vector<uint32_t> d_doc_offsets(n_anchors + 1);
  thrust::fill(thrust_policy, d_doc_offsets.begin(), d_doc_offsets.end(), 0);

  thrust::device_vector<uint32_t> anchor_keys(n_anchors);
  thrust::sequence(thrust_policy, anchor_keys.begin(), anchor_keys.end());

  thrust::device_vector<uint32_t> start_positions(n_anchors);
  thrust::lower_bound(thrust_policy,
    anchor_ids.begin(), anchor_ids.end(),
    anchor_keys.begin(), anchor_keys.end(),
    start_positions.begin());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in lower_bound: " + std::string(cudaGetErrorString(err)));
  }

  thrust::copy(thrust_policy, start_positions.begin(), start_positions.end(), d_doc_offsets.begin());
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in device-to-device copy: " + std::string(cudaGetErrorString(err)));
  }
  d_doc_offsets[n_anchors] = unique_count;

  unique_doc_ids.resize(unique_count);
  doc_offsets.resize(n_anchors + 1);

  auto stream = raft::resource::get_cuda_stream(handle);

  raft::copy(unique_doc_ids.data(),
             thrust::raw_pointer_cast(d_unique_doc_ids.data()),
             unique_count,
             stream);
  raft::resource::sync_stream(handle);

  raft::copy(doc_offsets.data(),
             thrust::raw_pointer_cast(d_doc_offsets.data()),
             n_anchors + 1,
             stream);
  raft::resource::sync_stream(handle);
}

} // namespace vecflow_chamfer::detail
