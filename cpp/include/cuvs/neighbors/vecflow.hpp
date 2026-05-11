/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace cuvs::neighbors::vecflow {

/**
 * @brief The vecflow index holds all internal information required for search.
 *
 * This includes the IVF-graph index, the IVF-BFS index, and metadata such as label sizes,
 * offsets, and the mapping of data points for each label.
 *
 */
template <typename data_t>
struct index {
  cuvs::neighbors::cagra::index<data_t, uint32_t> ivf_graph_index;
  cuvs::neighbors::ivf_flat::index<data_t, int64_t> ivf_bfs_index;

  int specificity_threshold = 2000;

  // Per-label CSR metadata for the IVF-Graph (CAGRA) lane.
  //   graph_index_map     : flat permutation of source row ids; concatenated
  //                         per-label posting lists.
  //   graph_label_size[l] : number of points carrying label `l` in this lane.
  //   graph_label_offset[l]: prefix sum of `graph_label_size`.
  raft::device_vector<uint32_t, int64_t> graph_index_map;
  raft::device_vector<uint32_t, int64_t> graph_label_size;
  raft::device_vector<uint32_t, int64_t> graph_label_offset;
  // Per-label size for the IVF-BFS lane (offsets are reconstructed at search time).
  raft::device_vector<uint32_t, int64_t> bfs_label_size;
  // Number of dataset points carrying each label (i.e. unnormalized
  // specificity). Indexed by label id. Used to decide which lane a query
  // routes to (vs `specificity_threshold`) and to pick the primary label
  // for multi-label AND queries.
  raft::device_vector<uint32_t, int64_t> label_freq;

  // Multi-label AND inline filter buffers — populated only when
  // `multi_label=true` was passed to `vecflow::build()`. Empty (size 0)
  // otherwise; the regular single-label `vecflow::search()` does not
  // touch them. Layout:
  //   dataset_labels[ offsets[i] .. offsets[i+1] )  —  point i's labels,
  //                                                    sorted ascending.
  //   offsets length = n_dataset + 1.
  raft::device_vector<uint32_t, int64_t> dataset_labels;
  raft::device_vector<int64_t,  int64_t> dataset_label_offsets;
};

/**
 * @brief Builds (or loads) the vecflow index.
 *
 * This function builds (or loads from file) both the IVF-graph index and the IVF-BFS index.
 *
 * @param res                   RAFT shared resources.
 * @param d_dataset             Device matrix view of the dataset.
 * @param data_label_vecs       Vector of vectors of data labels.
 * @param graph_degree          Desired graph degree.
 * @param specificity_threshold Threshold to decide which labels go to CAGRA vs. BFS.
 * @param graph_fname           (Optional) File name to load/save the IVF-graph index.
 * @param bfs_fname             (Optional) File name to load/save the BFS index.
 * @param force_rebuild         (Optional) Whether to force rebuild the index.
 * @param multi_label           (Optional, default `false`) If true, additionally prepare
 *                              the CSR `dataset_labels` + `dataset_label_offsets` buffers
 *                              on the returned index so that `search_multi_labels()` can
 *                              run AND-mode queries. Adds a one-time pass over
 *                              `data_label_vecs` at build time; leaves zero overhead on
 *                              the single-label `search()` path.
 */
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const float, int64_t> d_dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "",
           bool force_rebuild = false,
           bool multi_label = false) -> cuvs::neighbors::vecflow::index<float>;

auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const int8_t, int64_t> d_dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "",
           bool force_rebuild = false,
           bool multi_label = false) -> cuvs::neighbors::vecflow::index<int8_t>;
/**
 * @brief Performs a vecflow search.
 *
 * Given a set of queries and query labels, this function searches for the top-k nearest
 * neighbors using the vecflow index.
 *
 * @param res           RAFT shared resources.
 * @param index         The vecflow index to use.
 * @param queries       Device matrix view of query vectors.
 * @param query_labels  Device vector view of query labels.
 * @param itopk_size    Number of top results to return.
 * @param neighbors     [out] Device matrix view to hold neighbor indices.
 * @param distances     [out] Device matrix view to hold distances.
 */
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<float>& index,
            raft::device_matrix_view<const float, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<int8_t>& index,
            raft::device_matrix_view<const int8_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

/**
 * @brief Multi-label AND search (2 labels per query).
 *
 * Each query carries two labels; a dataset point is considered a candidate
 * only if it contains BOTH labels. The function looks up `index.label_freq` to
 * automatically determine which of the two labels has the larger specificity
 * (i.e. is more common); that one becomes the "primary" label used to select
 * the IVF posting list, and the other is checked inline by the CAGRA / BFS
 * kernels. Caller may pass `query_labels_a` and `query_labels_b` in any
 * order — no pre-sorting required.
 *
 * Requires the index to have been built with `multi_label=true` so that
 * `index.dataset_labels` and `index.dataset_label_offsets` are populated;
 * calling otherwise throws.
 *
 * @param res           RAFT shared resources.
 * @param index         VecFlow index built with `multi_label=true`.
 * @param queries       Device matrix view of query vectors [n_q, dim].
 * @param query_labels_a First label per query [n_q]; order vs `_b` is irrelevant.
 * @param query_labels_b Second label per query [n_q].
 * @param itopk_size    Internal top-k buffer size for CAGRA search.
 * @param neighbors     [out] [n_q, topk] neighbor indices.
 * @param distances     [out] [n_q, topk] distances.
 */
void search_multi_labels(shared_resources::configured_raft_resources& res,
                         cuvs::neighbors::vecflow::index<float>& index,
                         raft::device_matrix_view<const float, int64_t> queries,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_a,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_b,
                         int itopk_size,
                         raft::device_matrix_view<uint32_t, int64_t> neighbors,
                         raft::device_matrix_view<float, int64_t> distances);

void search_multi_labels(shared_resources::configured_raft_resources& res,
                         cuvs::neighbors::vecflow::index<int8_t>& index,
                         raft::device_matrix_view<const int8_t, int64_t> queries,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_a,
                         raft::device_vector_view<const uint32_t, int64_t> query_labels_b,
                         int itopk_size,
                         raft::device_matrix_view<uint32_t, int64_t> neighbors,
                         raft::device_matrix_view<float, int64_t> distances);

}  // namespace cuvs::neighbors::vecflow
