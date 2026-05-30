#include <vecflow_chamfer/io.hpp>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace vecflow_chamfer {

dataset load_dataset(const std::string& dataset_dir, const std::string& prefix) {
  dataset ds;

  std::string doc_embeddings_path = dataset_dir + "/" + prefix + ".doc.embeddings.fp16.fbin";
  std::string doc_offsets_path = dataset_dir + "/" + prefix + ".doc.offsets.bin";

  std::ifstream doc_offsets_file(doc_offsets_path, std::ios::binary);
  if (!doc_offsets_file.is_open()) {
    throw std::runtime_error("Failed to open document offsets file: " + doc_offsets_path);
  }
  doc_offsets_file.seekg(0, std::ios::end);
  size_t file_size = doc_offsets_file.tellg();
  doc_offsets_file.seekg(0, std::ios::beg);

  size_t num_offsets = file_size / sizeof(int32_t);
  ds.doc_offsets.resize(num_offsets);
  doc_offsets_file.read(reinterpret_cast<char*>(ds.doc_offsets.data()), file_size);
  doc_offsets_file.close();

  ds.num_docs = static_cast<int32_t>(num_offsets) - 1;
  ds.num_doc_tokens = ds.doc_offsets.back();

  std::ifstream doc_emb_file(doc_embeddings_path, std::ios::binary);
  if (!doc_emb_file.is_open()) {
    throw std::runtime_error("Failed to open document embeddings file: " + doc_embeddings_path);
  }
  doc_emb_file.seekg(0, std::ios::end);
  file_size = doc_emb_file.tellg();
  doc_emb_file.seekg(0, std::ios::beg);

  size_t num_elements = file_size / sizeof(__half);
  ds.embedding_dim = static_cast<int>(num_elements / ds.num_doc_tokens);

  // Pick the host-visible allocator based on what the GPU can reach. On
  // Grace-Hopper (and any device reporting pageable-memory access) a plain
  // malloc is GPU-readable through C2C/HMM/ATS, so we keep the buffer in
  // ordinary RAM and just advise PreferredLocation=host. On PCIe parts the GPU
  // can't follow pageable host pointers, so we fall back to cudaMallocManaged
  // with the same PreferredLocation=host advise plus AccessedBy=device — pages
  // stay resident on host RAM and the GPU maps them directly.
  int device_id = 0;
  cudaGetDevice(&device_id);
  int pageable_access = 0;
  cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, device_id);

  cudaMemLocation host_loc;
  host_loc.type = cudaMemLocationTypeHost;
  host_loc.id   = 0;

  if (pageable_access == 1) {
    ds.doc_embeddings = static_cast<__half*>(malloc(file_size));
    if (ds.doc_embeddings == nullptr) {
      throw std::runtime_error("Failed to allocate host memory for doc_embeddings");
    }
    ds.doc_embeddings_managed = false;
    cudaMemAdvise_v2(ds.doc_embeddings, file_size,
                     cudaMemAdviseSetPreferredLocation, host_loc);
  } else {
    void* p = nullptr;
    if (cudaMallocManaged(&p, file_size) != cudaSuccess || p == nullptr) {
      throw std::runtime_error("cudaMallocManaged failed for doc_embeddings");
    }
    ds.doc_embeddings = static_cast<__half*>(p);
    ds.doc_embeddings_managed = true;

    cudaMemLocation dev_loc;
    dev_loc.type = cudaMemLocationTypeDevice;
    dev_loc.id   = device_id;
    cudaMemAdvise_v2(p, file_size, cudaMemAdviseSetPreferredLocation, host_loc);
    cudaMemAdvise_v2(p, file_size, cudaMemAdviseSetAccessedBy,        dev_loc);
  }

  // Read directly into the host buffer (no staging copy).
  doc_emb_file.read(reinterpret_cast<char*>(ds.doc_embeddings), file_size);
  doc_emb_file.close();

  ds.doc_token_to_doc.resize(ds.num_doc_tokens);
  for (int32_t i = 0; i < ds.num_docs; i++) {
    for (uint32_t j = ds.doc_offsets[i]; j < ds.doc_offsets[i + 1]; j++) {
      ds.doc_token_to_doc[j] = i;
    }
  }

  return ds;
}

query_set load_queries(const std::string& dataset_dir, const std::string& prefix,
                       int embedding_dim, int num_tokens_per_query) {
  query_set queries;
  queries.embedding_dim = embedding_dim;
  queries.num_tokens_per_query = num_tokens_per_query;

  std::string query_embeddings_path = dataset_dir + "/" + prefix + ".query.embeddings.fp16.fbin";

  std::ifstream query_emb_file(query_embeddings_path, std::ios::binary);
  if (!query_emb_file.is_open()) {
    throw std::runtime_error("Failed to open query embeddings file: " + query_embeddings_path);
  }
  query_emb_file.seekg(0, std::ios::end);
  size_t file_size = query_emb_file.tellg();
  query_emb_file.seekg(0, std::ios::beg);

  size_t num_elements = file_size / sizeof(__half);
  queries.query_embeddings.resize(num_elements);
  query_emb_file.read(reinterpret_cast<char*>(queries.query_embeddings.data()), file_size);
  query_emb_file.close();

  queries.num_queries = static_cast<int32_t>(num_elements / (num_tokens_per_query * embedding_dim));

  return queries;
}

void release_dataset(dataset& ds) {
  if (ds.doc_embeddings != nullptr) {
    if (ds.doc_embeddings_managed) {
      cudaFree(ds.doc_embeddings);
    } else {
      free(ds.doc_embeddings);
    }
    ds.doc_embeddings = nullptr;
  }
  ds.doc_embeddings_managed = false;
}

} // namespace vecflow_chamfer
