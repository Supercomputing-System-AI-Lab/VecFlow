// Internal-only helpers for chamfer_core kernels. Not part of the public API.
#pragma once

#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Internal launch-config, tile, and shape constants. Not part of the public API
// — consumers use chamfer_core_score(...) / chamfer_core_score_anchor(...)
// which validate against these.
constexpr int CHAMFER_BLOCK_SIZE  = 256;
constexpr int TILE_COL_WIDTH      = 128;  // compiled-in embedding dimension
constexpr int Q_ROW               = 32;   // compiled-in number of query rows
constexpr int MAX_NUM_BI_SAMPLES  = 64;
constexpr int WMMA_M              = 32;
constexpr int WMMA_N              = 8;
constexpr int WMMA_K              = 16;
constexpr int WARP_SIZE           = 32;

template <typename T>
constexpr __host__ __device__ __forceinline__ int skew_dim(int ndim) {
  if constexpr (std::is_same<T, float>::value) {
    ndim = ((ndim + 3) / 4) * 4;
    return ndim + (ndim % 32 == 0) * 4;
  }
}

constexpr int SKEWED_MAX_NUM_BI_SAMPLES = skew_dim<float>(MAX_NUM_BI_SAMPLES);

__device__ __forceinline__ void load_optimized_vec(__half* vec_buffer,
                                                   const __half* d_vec,
                                                   const int load_dims,
                                                   const int num_rows,
                                                   const int padding_dims,
                                                   const int lane_id) {
  if ((size_t)d_vec % sizeof(float4) == 0 && (size_t)vec_buffer % sizeof(float4) == 0 &&
      load_dims % 8 == 0 && padding_dims % 8 == 0) {

    int row_id = lane_id / (load_dims / 8);
    int col_id = lane_id % (load_dims / 8);

    if (row_id < num_rows) {
      const float4* src_ptr = (const float4*)(d_vec + load_dims * row_id + col_id * 8);
      float4 loaded_data = __ldg(src_ptr);
      *(float4*)(vec_buffer + row_id * padding_dims + col_id * 8) = loaded_data;
    }
  } else {
    constexpr int num_load_elems_per_warp = WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < num_rows; i++) {
      for (int step = 0; step < (load_dims + num_load_elems_per_warp - 1) / num_load_elems_per_warp; step++) {
        int idx = step * num_load_elems_per_warp + lane_id;
        if (idx < load_dims) {
          vec_buffer[i * padding_dims + idx] = __ldg(&d_vec[i * load_dims + idx]);
        } else if (idx < padding_dims) {
          vec_buffer[i * padding_dims + idx] = __float2half(0.0f);
        }
      }
    }
  }
}

__device__ __forceinline__ void load_two_vec(__half* vec_buffer,
                                             const __half* d1_vec,
                                             const __half* d2_vec,
                                             const int load_dims,
                                             const int num_rows,
                                             const int padding_dims,
                                             const int lane_id) {
  if ((size_t)d1_vec % sizeof(float4) == 0 && (size_t)d2_vec % sizeof(float4) == 0 &&
      (size_t)vec_buffer % sizeof(float4) == 0 && load_dims % 8 == 0 && padding_dims % 8 == 0) {

    int row_id = lane_id / (load_dims / 8);
    int col_id = lane_id % (load_dims / 8);

    if (row_id < num_rows) {
      const __half* src_vec = (row_id == 0) ? d1_vec : d2_vec;
      const int4* src_ptr = reinterpret_cast<const int4*>(src_vec + col_id * 8);
      int4 loaded_data = __ldg(src_ptr);
      *reinterpret_cast<int4*>(vec_buffer + row_id * padding_dims + col_id * 8) = loaded_data;
    }
  } else {
    constexpr int num_load_elems_per_warp = WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < num_rows; i++) {
      for (int step = 0; step < (load_dims + num_load_elems_per_warp - 1) / num_load_elems_per_warp; step++) {
        int idx = step * num_load_elems_per_warp + lane_id;
        if (i == 0) {
          vec_buffer[i * padding_dims + idx] = __ldg(&d1_vec[idx]);
        } else {
          vec_buffer[i * padding_dims + idx] = __ldg(&d2_vec[idx]);
        }
      }
    }
  }
  __syncwarp();
}

// Reconstruct a single doc token from VPQ side-data into `output` (shared mem).
//
//   token ≈ vq_codebook[vq_label]  +  pq_decode(pq_codes)
//
// The VQ side reuses the MaxIVF anchor centroids (fp16). The PQ side is the
// cuVS-trained, use_subspaces=true codebook: a row-major matrix of shape
// [pq_dim * (1<<pq_bits), pq_len] in fp32, where subspace j owns rows
// [j*n_centers, (j+1)*n_centers). One full warp cooperates to fill `output`;
// `lane_id` is the caller's threadIdx.x % WARP_SIZE.
//
// Supports pq_bits in [4, 8]. Columns [dim, TILE_COL_WIDTH) of `output` are
// zero-filled so WMMA sees clean padding.
__device__ __forceinline__ void load_vpq_residual(
    const __half* __restrict__ vq_codebook,
    const float*  __restrict__ pq_codebook,
    const uint8_t* __restrict__ pq_codes,
    uint32_t vq_label,
    __half* output,
    int dim,
    uint32_t pq_bits,
    uint32_t pq_dim,
    uint32_t pq_len,
    int lane_id)
{
  const uint32_t n_centers = 1u << pq_bits;
  const __half* vq_base_ptr =
    vq_codebook + static_cast<uint64_t>(vq_label) * static_cast<uint64_t>(dim);

  for (int d = lane_id; d < dim; d += WARP_SIZE) {
    __half vq_val = __ldg(&vq_base_ptr[d]);

    const uint32_t subspace        = static_cast<uint32_t>(d) / pq_len;
    const uint32_t dim_in_subspace = static_cast<uint32_t>(d) % pq_len;

    uint32_t pq_code;
    if (pq_bits == 8) {
      pq_code = pq_codes[subspace];
    } else {
      const uint32_t bit_offset = subspace * pq_bits;
      const uint32_t byte_idx   = bit_offset >> 3;
      const uint32_t bit_shift  = bit_offset & 7u;
      const uint32_t lo = pq_codes[byte_idx];
      const uint32_t hi = (bit_shift + pq_bits > 8u) ? pq_codes[byte_idx + 1] : 0u;
      pq_code = (((hi << 8) | lo) >> bit_shift) & ((1u << pq_bits) - 1u);
    }

    const float pq_val_f =
      __ldg(&pq_codebook[(subspace * n_centers + pq_code) * pq_len + dim_in_subspace]);
    output[d] = __hadd(vq_val, __float2half(pq_val_f));
  }

  for (int d = dim + lane_id; d < TILE_COL_WIDTH; d += WARP_SIZE) {
    output[d] = __float2half(0.0f);
  }
}

__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int;
  int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(max(__int_as_float(assumed), val)));
  } while (assumed != old);
  return __int_as_float(old);
}

#if __CUDA_ARCH__ >= 700
  #define PREFETCH_GLOBAL(addr) asm volatile ("prefetch.global.L2 [%0];" :: "l"(addr) : "memory")
#else
  #define PREFETCH_GLOBAL(addr)
#endif
