#include <vecflow_chamfer/search.hpp>
#include <vecflow_chamfer/kernels.hpp>
#include "find_unique_ids.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map.cuh>
#include <cuvs/neighbors/cagra.hpp>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

namespace vecflow_chamfer {

namespace {

// Fill the output matrices with sentinels so rows that produce fewer than
// final_topk hits are unambiguously "no result" entries.
void fill_output_sentinels(raft::device_resources const& res,
                           raft::device_matrix_view<uint32_t, int64_t> neighbors,
                           raft::device_matrix_view<float,    int64_t> distances) {
  raft::linalg::map_offset(res, neighbors,
                           [] __device__(int64_t) { return uint32_t{0xFFFFFFFFu}; });
  raft::linalg::map_offset(res, distances,
                           [] __device__(int64_t) { return -FLT_MAX; });
}

void search_impl(raft::device_resources const& res,
                 search_params const& params,
                 index   const& idx,
                 dataset const& ds,
                 query_set const& queries,
                 raft::device_matrix_view<uint32_t, int64_t> neighbors,
                 raft::device_matrix_view<float,    int64_t> distances,
                 search_stats* stats) {
  using clock = std::chrono::high_resolution_clock;
  using ms = std::chrono::duration<double, std::milli>;

  const uint32_t n_queries = queries.num_queries;
  const uint32_t n_tokens_per_query = queries.num_tokens_per_query;
  const uint32_t dim = queries.embedding_dim;
  const uint32_t final_topk = params.final_topk;

  const bool use_vpq = params.use_vpq && idx.vpq.has_value();
  if (params.use_vpq && !idx.vpq.has_value()) {
    throw std::invalid_argument(
      "search: use_vpq=true requires an index built with index_params.pq_bits > 0");
  }
  const uint32_t pq_bits           = use_vpq ? idx.vpq->pq_bits : 0u;
  const uint32_t pq_dim_v          = use_vpq ? idx.vpq->pq_dim  : 0u;
  const uint32_t pq_len            = use_vpq ? idx.vpq->pq_len  : 0u;
  const uint32_t encoded_row_length =
    use_vpq ? static_cast<uint32_t>(idx.vpq->codes.extent(1)) : 0u;

  auto stream = raft::resource::get_cuda_stream(res);

  fill_output_sentinels(res, neighbors, distances);

  if (stats != nullptr) {
    stats->first_stage_unique_doc_ids.assign(n_queries, std::vector<uint32_t>{});
  }

  double total_candidate_generation = 0.0;
  double total_unique_id_finding = 0.0;
  double total_anchor_chamfer_reranking = 0.0;
  double total_first_sorting = 0.0;
  double total_vpq_reranking = 0.0;
  double total_vpq_sorting = 0.0;
  double total_full_chamfer_reranking = 0.0;
  double total_second_sorting = 0.0;
  double total_candidates = 0.0;
  double total_stage4_input = 0.0;
  double total_stage4_output = 0.0;
  double total_stage_vpq_output = 0.0;
  double total_stage6_output = 0.0;

  cuvs::neighbors::cagra::search_params cagra_search_params;
  cagra_search_params.itopk_size = params.itopk;
  cagra_search_params.algo = cuvs::neighbors::cagra::search_algo::MULTI_CTA;

  // Upload all queries to device once (not timed).
  auto d_queries = raft::make_device_matrix<half, int64_t>(
    res, static_cast<int64_t>(n_queries) * n_tokens_per_query, dim);
  raft::copy(d_queries.data_handle(), queries.query_embeddings.data(),
             queries.query_embeddings.size(), stream);
  raft::resource::sync_stream(res);

  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(
    res, n_tokens_per_query, params.search_topk);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(
    res, n_tokens_per_query, params.search_topk);

  auto hash_table = raft::make_device_matrix<uint32_t, int64_t>(res, 1, GLOBAL_HASH_TABLE_SIZE);
  std::vector<uint32_t> h_hash_table(GLOBAL_HASH_TABLE_SIZE, 0xFFFFFFFF);

  raft::resource::sync_stream(res);
  auto t_search_start = clock::now();

  for (uint32_t q_idx = 0; q_idx < n_queries; q_idx++) {
    // Reset hash table for each query (not timed; sync before timer starts).
    raft::copy(hash_table.data_handle(), h_hash_table.data(), GLOBAL_HASH_TABLE_SIZE, stream);
    raft::resource::sync_stream(res);
    auto t_prev = clock::now();

    // Stage 1: CAGRA candidate generation against anchor index.
    auto query_batch = raft::make_device_matrix_view<half, int64_t>(
      d_queries.data_handle() + static_cast<int64_t>(q_idx) * n_tokens_per_query * dim,
      n_tokens_per_query, dim);
    cuvs::neighbors::cagra::search(res, cagra_search_params, idx.anchor_cagra_index,
                                   raft::make_const_mdspan(query_batch),
                                   cagra_neighbors.view(), cagra_distances.view());
    raft::resource::sync_stream(res);
    auto t_s1 = clock::now();
    total_candidate_generation += ms(t_s1 - t_prev).count();
    t_prev = t_s1;

    // Stage 2: unique doc IDs via global hash table.
    auto unique_counter = raft::make_device_vector<int, int64_t>(res, 1);
    int h_n_doc_ids = 0;
    auto doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, GLOBAL_HASH_TABLE_SIZE);
    raft::copy(unique_counter.data_handle(), &h_n_doc_ids, 1, stream);
    int num_neighbors = n_tokens_per_query * params.search_topk;
    detail::find_unique_doc_ids(
      cagra_neighbors.data_handle(), idx.anchor_to_doc_ids.data_handle(),
      idx.anchor_to_doc_ids_offsets.data_handle(), hash_table.data_handle(),
      num_neighbors, doc_ids.data_handle(), unique_counter.data_handle());
    raft::resource::sync_stream(res);
    raft::copy(&h_n_doc_ids, unique_counter.data_handle(), 1, stream);
    auto unique_doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, h_n_doc_ids);
    raft::copy(unique_doc_ids.data_handle(), doc_ids.data_handle(), h_n_doc_ids, stream);
    raft::resource::sync_stream(res);
    auto t_s2 = clock::now();
    total_unique_id_finding += ms(t_s2 - t_prev).count();
    t_prev = t_s2;
    total_candidates += h_n_doc_ids;
    total_stage4_input += h_n_doc_ids;

    if (stats != nullptr && h_n_doc_ids > 0) {
      std::vector<uint32_t> h_unique_doc_ids(h_n_doc_ids);
      raft::copy(h_unique_doc_ids.data(), unique_doc_ids.data_handle(), h_n_doc_ids, stream);
      raft::resource::sync_stream(res);
      stats->first_stage_unique_doc_ids[q_idx] = std::move(h_unique_doc_ids);
    }

    // Stage 3: anchor-Chamfer proxy filter.
    auto anchor_chamfer_scores = raft::make_device_vector<float, int64_t>(res, unique_doc_ids.size());
    kernels::chamfer_score_anchor(
      static_cast<uint32_t>(unique_doc_ids.size()),
      n_tokens_per_query, dim,
      query_batch.data_handle(),
      idx.anchor_cagra_index.dataset().data_handle(),
      unique_doc_ids.data_handle(),
      idx.doc_offsets.data_handle(),
      idx.anchor_labels.data_handle(),
      anchor_chamfer_scores.data_handle(),
      stream);
    raft::resource::sync_stream(res);
    auto t_s3 = clock::now();
    total_anchor_chamfer_reranking += ms(t_s3 - t_prev).count();
    t_prev = t_s3;

    // Stage 4: top-(refine_rate * final_topk) candidates by proxy score.
    const size_t refined_k = std::min(
      static_cast<size_t>(params.refine_rate * final_topk),
      static_cast<size_t>(anchor_chamfer_scores.size())
    );
    auto temp_anchor_scores = raft::make_device_vector<float, int64_t>(res, anchor_chamfer_scores.size());
    auto temp_anchor_ids = raft::make_device_vector<uint32_t, int64_t>(res, anchor_chamfer_scores.size());
    raft::copy(temp_anchor_scores.data_handle(), anchor_chamfer_scores.data_handle(),
               anchor_chamfer_scores.size(), stream);
    raft::copy(temp_anchor_ids.data_handle(), unique_doc_ids.data_handle(), unique_doc_ids.size(), stream);
    thrust::sort_by_key(raft::resource::get_thrust_policy(res),
                        temp_anchor_scores.data_handle(),
                        temp_anchor_scores.data_handle() + static_cast<size_t>(anchor_chamfer_scores.size()),
                        temp_anchor_ids.data_handle(),
                        thrust::greater<float>());

    auto refined_doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, refined_k);
    if (refined_k > 0) {
      raft::copy(refined_doc_ids.data_handle(), temp_anchor_ids.data_handle(), refined_k, stream);
    }
    raft::resource::sync_stream(res);
    auto t_s4 = clock::now();
    total_first_sorting += ms(t_s4 - t_prev).count();
    t_prev = t_s4;
    total_stage4_output += refined_k;

    // Stage 5/6 (optional): VPQ chamfer rerank + top-K narrowing. Slots
    // between the anchor proxy and the full rerank. The VPQ scores are an
    // approximation of full Chamfer but cheap (1 byte per subspace), so this
    // pre-filters before the more expensive full-precision rerank.
    const uint32_t* rerank_input_ids_ptr = refined_doc_ids.data_handle();
    size_t rerank_input_k = refined_k;
    auto vpq_refined_doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, 0);
    if (use_vpq && refined_k > 0) {
      auto vpq_scores = raft::make_device_vector<float, int64_t>(res, refined_k);
      kernels::chamfer_score_vpq(
        static_cast<uint32_t>(refined_k),
        n_tokens_per_query, dim,
        query_batch.data_handle(),
        idx.anchor_cagra_index.dataset().data_handle(),
        idx.vpq->pq_codebook.data_handle(),
        idx.vpq->codes.data_handle(),
        idx.anchor_labels.data_handle(),
        refined_doc_ids.data_handle(),
        idx.doc_offsets.data_handle(),
        encoded_row_length,
        pq_bits, pq_dim_v, pq_len,
        vpq_scores.data_handle(),
        stream);
      raft::resource::sync_stream(res);
      auto t_s5_vpq = clock::now();
      total_vpq_reranking += ms(t_s5_vpq - t_prev).count();
      t_prev = t_s5_vpq;

      const size_t vpq_refined_k = std::min(
        static_cast<size_t>(params.refine_rate_vpq * final_topk),
        refined_k);

      auto temp_vpq_scores = raft::make_device_vector<float, int64_t>(res, refined_k);
      auto temp_vpq_ids    = raft::make_device_vector<uint32_t, int64_t>(res, refined_k);
      raft::copy(temp_vpq_scores.data_handle(), vpq_scores.data_handle(), refined_k, stream);
      raft::copy(temp_vpq_ids.data_handle(),    refined_doc_ids.data_handle(), refined_k, stream);
      thrust::sort_by_key(raft::resource::get_thrust_policy(res),
                          temp_vpq_scores.data_handle(),
                          temp_vpq_scores.data_handle() + refined_k,
                          temp_vpq_ids.data_handle(),
                          thrust::greater<float>());

      vpq_refined_doc_ids = raft::make_device_vector<uint32_t, int64_t>(res, vpq_refined_k);
      if (vpq_refined_k > 0) {
        raft::copy(vpq_refined_doc_ids.data_handle(),
                   temp_vpq_ids.data_handle(), vpq_refined_k, stream);
      }
      raft::resource::sync_stream(res);
      auto t_s6_vpq = clock::now();
      total_vpq_sorting += ms(t_s6_vpq - t_prev).count();
      t_prev = t_s6_vpq;

      rerank_input_ids_ptr = vpq_refined_doc_ids.data_handle();
      rerank_input_k       = vpq_refined_k;
      total_stage_vpq_output += vpq_refined_k;
    }

    // Stage 7 (or 5 when VPQ disabled): full-precision Chamfer rerank.
    auto full_chamfer_scores = raft::make_device_vector<float, int64_t>(res, rerank_input_k);
    kernels::chamfer_score(
      static_cast<uint32_t>(rerank_input_k),
      n_tokens_per_query, dim,
      query_batch.data_handle(),
      ds.doc_embeddings,
      rerank_input_ids_ptr,
      idx.doc_offsets.data_handle(),
      full_chamfer_scores.data_handle(),
      stream);
    raft::resource::sync_stream(res);
    auto t_s7 = clock::now();
    total_full_chamfer_reranking += ms(t_s7 - t_prev).count();
    t_prev = t_s7;

    // Stage 8: final top-K. Sort the refined candidates and write the top
    // `final_k` entries straight into row `q_idx` of the output matrices.
    auto temp_full_scores = raft::make_device_vector<float, int64_t>(res, full_chamfer_scores.size());
    auto temp_full_ids = raft::make_device_vector<uint32_t, int64_t>(res, full_chamfer_scores.size());
    raft::copy(temp_full_scores.data_handle(), full_chamfer_scores.data_handle(),
               full_chamfer_scores.size(), stream);
    raft::copy(temp_full_ids.data_handle(), rerank_input_ids_ptr, rerank_input_k, stream);
    thrust::sort_by_key(raft::resource::get_thrust_policy(res),
                        temp_full_scores.data_handle(),
                        temp_full_scores.data_handle() + static_cast<size_t>(full_chamfer_scores.size()),
                        temp_full_ids.data_handle(),
                        thrust::greater<float>());

    const size_t final_k = std::min(static_cast<size_t>(final_topk),
                                    static_cast<size_t>(full_chamfer_scores.size()));
    if (final_k > 0) {
      uint32_t* row_neighbors = neighbors.data_handle() + static_cast<int64_t>(q_idx) * final_topk;
      float*    row_distances = distances.data_handle() + static_cast<int64_t>(q_idx) * final_topk;
      raft::copy(row_neighbors, temp_full_ids.data_handle(),    final_k, stream);
      raft::copy(row_distances, temp_full_scores.data_handle(), final_k, stream);
    }
    raft::resource::sync_stream(res);
    auto t_s8 = clock::now();
    total_second_sorting += ms(t_s8 - t_prev).count();
    total_stage6_output += final_k;
  }

  raft::resource::sync_stream(res);
  auto t_search_end = clock::now();
  const double wall_ms = ms(t_search_end - t_search_start).count();

  if (stats != nullptr) {
    stats->avg_candidate_generation_ms = total_candidate_generation / n_queries;
    stats->avg_unique_id_finding_ms = total_unique_id_finding / n_queries;
    stats->avg_anchor_chamfer_reranking_ms = total_anchor_chamfer_reranking / n_queries;
    stats->avg_first_sorting_ms = total_first_sorting / n_queries;
    stats->avg_vpq_reranking_ms = total_vpq_reranking / n_queries;
    stats->avg_vpq_sorting_ms   = total_vpq_sorting   / n_queries;
    stats->avg_full_chamfer_reranking_ms = total_full_chamfer_reranking / n_queries;
    stats->avg_second_sorting_ms = total_second_sorting / n_queries;
    stats->avg_candidates_per_query = total_candidates / n_queries;
    stats->total_wall_ms = wall_ms;
    stats->avg_per_query_ms = wall_ms / n_queries;
    stats->throughput_qps = (wall_ms > 0.0) ? (n_queries * 1000.0 / wall_ms) : 0.0;
    stats->avg_stage4_input_count = total_stage4_input / n_queries;
    stats->avg_stage4_output_count = total_stage4_output / n_queries;
    stats->avg_stage_vpq_output_count = total_stage_vpq_output / n_queries;
    stats->avg_stage6_output_count = total_stage6_output / n_queries;
  }
}

} // namespace

void search(raft::device_resources const& res,
            search_params const& params,
            index   const& idx,
            dataset const& ds,
            query_set const& queries,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float,    int64_t> distances) {
  search_impl(res, params, idx, ds, queries, neighbors, distances, nullptr);
}

void search(raft::device_resources const& res,
            search_params const& params,
            index   const& idx,
            dataset const& ds,
            query_set const& queries,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float,    int64_t> distances,
            search_stats& stats) {
  search_impl(res, params, idx, ds, queries, neighbors, distances, &stats);
}

} // namespace vecflow_chamfer
