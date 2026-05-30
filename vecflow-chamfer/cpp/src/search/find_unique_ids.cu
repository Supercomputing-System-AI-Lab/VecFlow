#include "find_unique_ids.hpp"

#include <climits>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ uint32_t hash_function_global(uint32_t key) {
  key ^= key >> 16;
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key & (GLOBAL_HASH_TABLE_SIZE - 1);
}

__global__ __launch_bounds__(HASH_BLOCK_SIZE, 4)
void find_unique_doc_ids_multi_cta_kernel(
  const uint32_t* __restrict__ sample_ids,
  const uint32_t* __restrict__ sample_to_doc_ids,
  const uint32_t* __restrict__ sample_to_doc_id_offset,
  int num_samples,
  uint32_t* __restrict__ global_hash_table,
  uint32_t* __restrict__ output,
  int* __restrict__ unique_counter) {

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int warp_id = tid / HASH_WARP_SIZE;
  int lane_id = tid % HASH_WARP_SIZE;

  int global_warp_id = bid * WARPS_PER_BLOCK + warp_id;

  if (global_warp_id >= num_samples) return;

  uint32_t sample_id = sample_ids[global_warp_id];
  uint32_t start_offset = sample_to_doc_id_offset[sample_id];
  uint32_t end_offset = sample_to_doc_id_offset[sample_id + 1];

  for (uint32_t doc_idx = start_offset + lane_id; doc_idx < end_offset; doc_idx += HASH_WARP_SIZE) {
    uint32_t doc_id = sample_to_doc_ids[doc_idx];

    uint32_t hash_idx = hash_function_global(doc_id);
    bool inserted = false;

    for (int probe = 0; probe < MAX_PROBE_DISTANCE && !inserted; probe++) {
      uint32_t probe_idx = (hash_idx + probe) & (GLOBAL_HASH_TABLE_SIZE - 1);
      uint32_t old_val = atomicCAS(&global_hash_table[probe_idx], UINT_MAX, doc_id);

      if (old_val == UINT_MAX || old_val == doc_id) {
        inserted = true;
      }
    }
  }
}

__global__ void extract_results_kernel(
  const uint32_t* __restrict__ global_hash_table,
  uint32_t* __restrict__ output,
  int* __restrict__ unique_counter) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < GLOBAL_HASH_TABLE_SIZE; i += stride) {
    if (global_hash_table[i] != UINT_MAX) {
      int pos = atomicAdd(unique_counter, 1);
      output[pos] = global_hash_table[i];
    }
  }
}

} // namespace

namespace vecflow_chamfer::detail {

void find_unique_doc_ids(const uint32_t* sample_ids,
                         const uint32_t* sample_to_doc_ids,
                         const uint32_t* sample_to_doc_id_offset,
                         const uint32_t* global_hash_table,
                         int num_samples,
                         uint32_t* output,
                         int* unique_counter) {
  int num_blocks = (num_samples + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  num_blocks = max(1, num_blocks);

  dim3 grid(num_blocks);
  dim3 block(HASH_BLOCK_SIZE);

  find_unique_doc_ids_multi_cta_kernel<<<grid, block>>>(
    sample_ids, sample_to_doc_ids, sample_to_doc_id_offset,
    num_samples, const_cast<uint32_t*>(global_hash_table), output, unique_counter);

  dim3 extract_grid(GLOBAL_HASH_TABLE_SIZE / HASH_BLOCK_SIZE);
  dim3 extract_block(HASH_BLOCK_SIZE);
  extract_results_kernel<<<extract_grid, extract_block>>>(
    global_hash_table, output, unique_counter);
}

} // namespace vecflow_chamfer::detail
