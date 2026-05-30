#pragma once

#include <string>

#include <vecflow_chamfer/types.hpp>

namespace vecflow_chamfer {

dataset   load_dataset(const std::string& dataset_dir, const std::string& prefix);

query_set load_queries(const std::string& dataset_dir, const std::string& prefix,
                       int embedding_dim, int num_tokens_per_query = 32);

// Frees doc_embeddings using the allocator chosen at load time (host malloc on
// HMM/ATS systems, cudaMallocManaged on PCIe). Safe to call on a default-
// constructed or already-released dataset.
void release_dataset(dataset& ds);

} // namespace vecflow_chamfer
