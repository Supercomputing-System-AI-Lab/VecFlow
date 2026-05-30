#include <vecflow_chamfer/kernels.hpp>
#include "chamfer_core_impl.cuh"

#include <cfloat>
#include <stdexcept>
#include <string>

#include <mma.h>

namespace vecflow_chamfer::kernels::detail {

template <typename Index_t>
__global__ __launch_bounds__(CHAMFER_BLOCK_SIZE, 4)
void chamfer_score_anchor(const __half* __restrict__ query,
                          const __half* __restrict__ dataset,
                          const uint32_t* __restrict__ doc_ids,
                          const uint32_t* __restrict__ doc_offsets,
                          const uint32_t* __restrict__ doc_token_to_rep,
                          float* __restrict__ chamfer_distances)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;

  // Each block processes a different B matrix
  int doc_id = doc_ids[blockIdx.x];
  int n = doc_offsets[doc_id + 1] - doc_offsets[doc_id];
  int doc_offset = doc_offsets[doc_id];

  // Pointer to the chamfer distance for this B matrix
  float* my_chamfer_distance = &chamfer_distances[blockIdx.x];

  constexpr int APAD = 8;
  constexpr int BPAD = 8;

  __shared__ __half s_av[Q_ROW][TILE_COL_WIDTH + APAD];
  __shared__ __half s_bv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];

  float* s_distances = (float*)&s_bv[0][0];

  __shared__ float s_row_max[Q_ROW];

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int num_warps = CHAMFER_BLOCK_SIZE / WARP_SIZE;

  constexpr int warps_y = Q_ROW / WMMA_M;
  int warp_id_y = warp_id % warps_y;
  int warp_id_x = warp_id / warps_y;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  if (threadIdx.x == 0) {
    *my_chamfer_distance = 0.0f;
  }

  #pragma unroll
  for (int i = threadIdx.x; i < Q_ROW; i += blockDim.x) {
    s_row_max[i] = -FLT_MAX;
  }

  #pragma unroll
  for (int i = 0; i < Q_ROW / (num_warps * 2); i++) {
    int row_idx = warp_id * 2 + i * (num_warps * 2);
    if (row_idx < Q_ROW) {
      size_t idx_in_data = row_idx * TILE_COL_WIDTH;
      load_optimized_vec(s_av[row_idx],
                         query + idx_in_data,
                         TILE_COL_WIDTH,
                         2,
                         TILE_COL_WIDTH + APAD,
                         lane_id);
    }
  }
  __syncthreads();

  const int tiles_n = (n + MAX_NUM_BI_SAMPLES - 1) / MAX_NUM_BI_SAMPLES;
  for (int tile_n = 0; tile_n < tiles_n; tile_n++) {
    int col_start = tile_n * MAX_NUM_BI_SAMPLES;
    int col_end = min(col_start + MAX_NUM_BI_SAMPLES, n);
    int cols_this_tile = col_end - col_start;

    wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / (num_warps * 2); i++) {
      int idx = warp_id * 2 + i * (num_warps * 2);
      int64_t idx_in_data = static_cast<int64_t>(doc_token_to_rep[doc_offset + col_start + idx]) * TILE_COL_WIDTH;
      int64_t idx_in_data_2 = static_cast<int64_t>(doc_token_to_rep[doc_offset + col_start + idx + 1]) * TILE_COL_WIDTH;
      if (idx + 1 < cols_this_tile) {
        load_two_vec(s_bv[idx],
                     dataset + idx_in_data,
                     dataset + idx_in_data_2,
                     TILE_COL_WIDTH,
                     2,
                     TILE_COL_WIDTH + BPAD,
                     lane_id);
      } else if (idx < cols_this_tile) {
        load_two_vec(s_bv[idx],
                     dataset + idx_in_data,
                     dataset + idx_in_data_2,
                     TILE_COL_WIDTH,
                     1,
                     TILE_COL_WIDTH + BPAD,
                     lane_id);
      }
    }
    __syncthreads();

    #pragma unroll 8
    for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
      if (warp_id_x < (cols_this_tile + WMMA_N - 1) / WMMA_N) {
        wmma::load_matrix_sync(a_frag, s_av[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(b_frag, s_bv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    __syncthreads();

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int row = warp_id; row < Q_ROW; row += num_warps) {
      float thread_max = -FLT_MAX;

      int col = lane_id;
      if (col < cols_this_tile) {
        thread_max = s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col];
      }

      col = lane_id + WARP_SIZE;
      if (col < cols_this_tile) {
        const float val = s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col];
        thread_max = max(thread_max, val);
      }

      #pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, thread_max, mask);
        thread_max = max(thread_max, other_max);
      }

      if (lane_id == 0) {
        atomicMaxFloat(&s_row_max[row], thread_max);
      }
    }

    __syncthreads();
  }

  if (warp_id == 0) {
    float warp_sum = 0.0f;
    for (int i = lane_id; i < Q_ROW; i += WARP_SIZE) {
      warp_sum += s_row_max[i];
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    if (lane_id == 0) {
      *my_chamfer_distance = warp_sum;
    }
    __syncthreads();
  }
#endif
}

template <typename Index_t>
__global__ __launch_bounds__(CHAMFER_BLOCK_SIZE, 4)
void chamfer_score(const __half* __restrict__ query,
                   const __half* __restrict__ dataset,
                   const uint32_t* __restrict__ doc_ids,
                   const uint32_t* __restrict__ doc_offsets,
                   float* __restrict__ chamfer_distances)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;

  int doc_id = doc_ids[blockIdx.x];
  int n = doc_offsets[doc_id + 1] - doc_offsets[doc_id];
  const __half* my_B = dataset + static_cast<int64_t>(doc_offsets[doc_id]) * TILE_COL_WIDTH;

  float* my_chamfer_distance = &chamfer_distances[blockIdx.x];

  constexpr int APAD = 8;
  constexpr int BPAD = 8;

  __shared__ __half s_av[Q_ROW][TILE_COL_WIDTH + APAD];
  __shared__ __half s_bv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];

  float* s_distances = (float*)&s_bv[0][0];

  __shared__ float s_row_max[Q_ROW];

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int num_warps = CHAMFER_BLOCK_SIZE / WARP_SIZE;

  constexpr int warps_y = Q_ROW / WMMA_M;
  int warp_id_y = warp_id % warps_y;
  int warp_id_x = warp_id / warps_y;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  if (threadIdx.x == 0) {
    *my_chamfer_distance = 0.0f;
  }

  #pragma unroll
  for (int i = threadIdx.x; i < Q_ROW; i += blockDim.x) {
    s_row_max[i] = -FLT_MAX;
  }
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < Q_ROW / (num_warps * 2); i++) {
    int row_idx = warp_id * 2 + i * (num_warps * 2);
    if (row_idx < Q_ROW) {
      size_t idx_in_data = static_cast<size_t>(row_idx) * TILE_COL_WIDTH;
      load_optimized_vec(s_av[row_idx],
                         query + idx_in_data,
                         TILE_COL_WIDTH,
                         2,
                         TILE_COL_WIDTH + APAD,
                         lane_id);
    }
  }
  __syncthreads();

  const int tiles_n = (n + MAX_NUM_BI_SAMPLES - 1) / MAX_NUM_BI_SAMPLES;

  for (int tile_n = 0; tile_n < tiles_n; tile_n++) {
    int col_start = tile_n * MAX_NUM_BI_SAMPLES;
    int col_end = (col_start + MAX_NUM_BI_SAMPLES > n) ? n : col_start + MAX_NUM_BI_SAMPLES;
    int cols_this_tile = col_end - col_start;

    wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / (num_warps * 2); i++) {
      int idx = warp_id * 2 + i * (num_warps * 2);
      int64_t idx_in_data = static_cast<int64_t>(col_start + idx) * TILE_COL_WIDTH;
      if (idx + 1 < cols_this_tile) {
        load_optimized_vec(s_bv[idx],
                           my_B + idx_in_data,
                           TILE_COL_WIDTH,
                           2,
                           TILE_COL_WIDTH + BPAD,
                           lane_id);
      } else if (idx < cols_this_tile) {
        load_optimized_vec(s_bv[idx],
                           my_B + idx_in_data,
                           TILE_COL_WIDTH,
                           1,
                           TILE_COL_WIDTH + BPAD,
                           lane_id);
      }
    }
    __syncthreads();

    if (tile_n < tiles_n - 1) {
      int next_col_start = (tile_n + 1) * MAX_NUM_BI_SAMPLES;
      int next_cols = min(MAX_NUM_BI_SAMPLES, n - next_col_start);

      if (threadIdx.x < 64) {
        int pf_col = threadIdx.x % next_cols;
        int pf_dim = (threadIdx.x / next_cols) * 16;

        while (pf_dim < TILE_COL_WIDTH) {
          PREFETCH_GLOBAL(my_B + static_cast<int64_t>(next_col_start + pf_col) * TILE_COL_WIDTH + pf_dim);
          pf_dim += 64;
        }
      }
    }

    #pragma unroll 8
    for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
      if (warp_id_x < (cols_this_tile + WMMA_N - 1) / WMMA_N) {
        wmma::load_matrix_sync(a_frag, s_av[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(b_frag, s_bv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    __syncthreads();

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int row = warp_id; row < Q_ROW; row += num_warps) {
      float thread_max = -FLT_MAX;

      int col = lane_id;
      if (col < cols_this_tile) {
        thread_max = s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col];
      }

      col = lane_id + WARP_SIZE;
      if (col < cols_this_tile) {
        const float val = s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col];
        thread_max = max(thread_max, val);
      }

      #pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, thread_max, mask);
        thread_max = max(thread_max, other_max);
      }

      if (lane_id == 0) {
        s_row_max[row] = max(s_row_max[row], thread_max);
      }
    }
    __syncthreads();
  }

  if (warp_id == 0) {
    float warp_sum = 0.0f;
    for (int i = lane_id; i < Q_ROW; i += WARP_SIZE) {
      warp_sum += s_row_max[i];
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    if (lane_id == 0) {
      *my_chamfer_distance = warp_sum;
    }
  }
#endif
}

template <typename Index_t>
__global__ __launch_bounds__(CHAMFER_BLOCK_SIZE, 4)
void chamfer_score_vpq(const __half* __restrict__ query,
                       const __half* __restrict__ vq_codebook,
                       const float*  __restrict__ pq_codebook,
                       const uint8_t* __restrict__ pq_codes,
                       const uint32_t* __restrict__ anchor_labels,
                       const uint32_t* __restrict__ doc_ids,
                       const uint32_t* __restrict__ doc_offsets,
                       uint32_t encoded_row_length,
                       uint32_t pq_bits,
                       uint32_t pq_dim,
                       uint32_t pq_len,
                       float* __restrict__ chamfer_distances)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;

  const int doc_id     = doc_ids[blockIdx.x];
  const int n          = doc_offsets[doc_id + 1] - doc_offsets[doc_id];
  const int doc_offset = doc_offsets[doc_id];

  float* my_chamfer_distance = &chamfer_distances[blockIdx.x];

  constexpr int APAD = 8;
  constexpr int BPAD = 8;

  __shared__ __half s_av[Q_ROW][TILE_COL_WIDTH + APAD];
  __shared__ __half s_bv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];

  float* s_distances = (float*)&s_bv[0][0];

  __shared__ float s_row_max[Q_ROW];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int num_warps = CHAMFER_BLOCK_SIZE / WARP_SIZE;

  constexpr int warps_y = Q_ROW / WMMA_M;
  const int warp_id_y = warp_id % warps_y;
  const int warp_id_x = warp_id / warps_y;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  if (threadIdx.x == 0) {
    *my_chamfer_distance = 0.0f;
  }

  #pragma unroll
  for (int i = threadIdx.x; i < Q_ROW; i += blockDim.x) {
    s_row_max[i] = -FLT_MAX;
  }

  #pragma unroll
  for (int i = 0; i < Q_ROW / (num_warps * 2); i++) {
    int row_idx = warp_id * 2 + i * (num_warps * 2);
    if (row_idx < Q_ROW) {
      size_t idx_in_data = static_cast<size_t>(row_idx) * TILE_COL_WIDTH;
      load_optimized_vec(s_av[row_idx],
                         query + idx_in_data,
                         TILE_COL_WIDTH,
                         2,
                         TILE_COL_WIDTH + APAD,
                         lane_id);
    }
  }
  __syncthreads();

  const int tiles_n = (n + MAX_NUM_BI_SAMPLES - 1) / MAX_NUM_BI_SAMPLES;
  for (int tile_n = 0; tile_n < tiles_n; tile_n++) {
    const int col_start      = tile_n * MAX_NUM_BI_SAMPLES;
    const int col_end        = (col_start + MAX_NUM_BI_SAMPLES > n) ? n : col_start + MAX_NUM_BI_SAMPLES;
    const int cols_this_tile = col_end - col_start;

    wmma::fill_fragment(c_frag, 0.0f);

    // Each warp reconstructs one doc token per inner iteration. With 8 warps
    // and MAX_NUM_BI_SAMPLES=64, that's 8 iterations to fill the tile.
    #pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      const int idx = warp_id + i * num_warps;
      if (idx < cols_this_tile) {
        const int64_t token_global =
          static_cast<int64_t>(doc_offset) + col_start + idx;
        const uint32_t vq_label = __ldg(&anchor_labels[token_global]);
        load_vpq_residual(vq_codebook,
                          pq_codebook,
                          pq_codes + token_global * static_cast<int64_t>(encoded_row_length),
                          vq_label,
                          s_bv[idx],
                          TILE_COL_WIDTH,
                          pq_bits, pq_dim, pq_len,
                          lane_id);
      }
    }
    __syncthreads();

    #pragma unroll 8
    for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
      if (warp_id_x < (cols_this_tile + WMMA_N - 1) / WMMA_N) {
        wmma::load_matrix_sync(a_frag, s_av[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(b_frag, s_bv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    __syncthreads();

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int row = warp_id; row < Q_ROW; row += num_warps) {
      float thread_max = -FLT_MAX;

      int col = lane_id;
      if (col < cols_this_tile) {
        thread_max = s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col];
      }

      col = lane_id + WARP_SIZE;
      if (col < cols_this_tile) {
        const float val = s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col];
        thread_max = max(thread_max, val);
      }

      #pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, thread_max, mask);
        thread_max = max(thread_max, other_max);
      }

      if (lane_id == 0) {
        atomicMaxFloat(&s_row_max[row], thread_max);
      }
    }

    __syncthreads();
  }

  if (warp_id == 0) {
    float warp_sum = 0.0f;
    for (int i = lane_id; i < Q_ROW; i += WARP_SIZE) {
      warp_sum += s_row_max[i];
    }
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    if (lane_id == 0) {
      *my_chamfer_distance = warp_sum;
    }
  }
#endif
}

// Explicit instantiations exported by libvecflow_chamfer_kernels.
template __global__ void chamfer_score_anchor<int>(
  const __half* __restrict__, const __half* __restrict__,
  const uint32_t* __restrict__, const uint32_t* __restrict__,
  const uint32_t* __restrict__, float* __restrict__);

template __global__ void chamfer_score<int>(
  const __half* __restrict__, const __half* __restrict__,
  const uint32_t* __restrict__, const uint32_t* __restrict__,
  float* __restrict__);

template __global__ void chamfer_score_vpq<int>(
  const __half* __restrict__, const __half* __restrict__,
  const float* __restrict__, const uint8_t* __restrict__,
  const uint32_t* __restrict__, const uint32_t* __restrict__,
  const uint32_t* __restrict__,
  uint32_t, uint32_t, uint32_t, uint32_t,
  float* __restrict__);

} // namespace vecflow_chamfer::kernels::detail

// ---- High-level launchers ---------------------------------------------------

namespace vecflow_chamfer::kernels {

static void validate_shape(uint32_t tokens_per_query, uint32_t dim) {
  if (static_cast<int>(tokens_per_query) != Q_ROW || static_cast<int>(dim) != TILE_COL_WIDTH) {
    throw std::invalid_argument(
      "chamfer kernel shape mismatch: kernel requires tokens_per_query=" +
      std::to_string(Q_ROW) + ", dim=" + std::to_string(TILE_COL_WIDTH) +
      "; got tokens_per_query=" + std::to_string(tokens_per_query) +
      ", dim=" + std::to_string(dim));
  }
}

void chamfer_score(uint32_t n_docs,
                   uint32_t tokens_per_query,
                   uint32_t dim,
                   const __half* query,
                   const __half* dataset,
                   const uint32_t* doc_ids,
                   const uint32_t* doc_offsets,
                   float* chamfer_distances,
                   cudaStream_t stream) {
  validate_shape(tokens_per_query, dim);
  if (n_docs == 0) return;
  dim3 grid(n_docs, 1, 1);
  dim3 block(CHAMFER_BLOCK_SIZE, 1, 1);
  detail::chamfer_score<int><<<grid, block, 0, stream>>>(
    query, dataset, doc_ids, doc_offsets, chamfer_distances);
}

void chamfer_score_anchor(uint32_t n_docs,
                          uint32_t tokens_per_query,
                          uint32_t dim,
                          const __half* query,
                          const __half* dataset,
                          const uint32_t* doc_ids,
                          const uint32_t* doc_offsets,
                          const uint32_t* doc_token_to_rep,
                          float* chamfer_distances,
                          cudaStream_t stream) {
  validate_shape(tokens_per_query, dim);
  if (n_docs == 0) return;
  dim3 grid(n_docs, 1, 1);
  dim3 block(CHAMFER_BLOCK_SIZE, 1, 1);
  detail::chamfer_score_anchor<int><<<grid, block, 0, stream>>>(
    query, dataset, doc_ids, doc_offsets, doc_token_to_rep, chamfer_distances);
}

void chamfer_score_vpq(uint32_t n_docs,
                       uint32_t tokens_per_query,
                       uint32_t dim,
                       const __half* query,
                       const __half* vq_codebook,
                       const float* pq_codebook,
                       const uint8_t* pq_codes,
                       const uint32_t* anchor_labels,
                       const uint32_t* doc_ids,
                       const uint32_t* doc_offsets,
                       uint32_t encoded_row_length,
                       uint32_t pq_bits,
                       uint32_t pq_dim,
                       uint32_t pq_len,
                       float* chamfer_distances,
                       cudaStream_t stream) {
  validate_shape(tokens_per_query, dim);
  if (pq_bits < 4 || pq_bits > 8) {
    throw std::invalid_argument(
      "chamfer_score_vpq: pq_bits must be in [4,8] (got " +
      std::to_string(pq_bits) + ")");
  }
  if (pq_dim == 0 || pq_len == 0 || pq_dim * pq_len != dim) {
    throw std::invalid_argument(
      "chamfer_score_vpq: pq_dim*pq_len must equal dim");
  }
  if (n_docs == 0) return;
  dim3 grid(n_docs, 1, 1);
  dim3 block(CHAMFER_BLOCK_SIZE, 1, 1);
  detail::chamfer_score_vpq<int><<<grid, block, 0, stream>>>(
    query, vq_codebook, pq_codebook, pq_codes, anchor_labels,
    doc_ids, doc_offsets, encoded_row_length,
    pq_bits, pq_dim, pq_len, chamfer_distances);
}

} // namespace vecflow_chamfer::kernels
