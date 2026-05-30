// Internal: hash-table dedup for the candidate-doc-IDs gathered from per-token
// CAGRA hits. Not part of the public API.
#pragma once

#include <cstdint>

#define HASH_BLOCK_SIZE 256
#define GLOBAL_HASH_TABLE_SIZE (2 << 16)
#define MAX_PROBE_DISTANCE 128
#define HASH_WARP_SIZE 32
#define WARPS_PER_BLOCK (HASH_BLOCK_SIZE / HASH_WARP_SIZE)

namespace vecflow_chamfer::detail {

// Launch: warp-per-sample multi-CTA hash + extract pass.
// global_hash_table must be pre-filled with 0xFFFFFFFF and sized GLOBAL_HASH_TABLE_SIZE.
// unique_counter must be pre-zeroed.
void find_unique_doc_ids(const uint32_t* sample_ids,
                         const uint32_t* sample_to_doc_ids,
                         const uint32_t* sample_to_doc_id_offset,
                         const uint32_t* global_hash_table,
                         int num_samples,
                         uint32_t* output,
                         int* unique_counter);

} // namespace vecflow_chamfer::detail
