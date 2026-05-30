// Direct usage of the public chamfer_score kernel.
//
// Generates a random fp16 query (tokens_per_query x dim) and a random fp16
// dataset (num_docs x tokens_per_doc x dim), then calls chamfer_core_score
// once per run. The library picks the kernel launch configuration internally;
// the result for each doc is the exact Chamfer score:
//
//   sum_{i in [0, tokens_per_query)}  max_{j in [0, tokens_per_doc)}  <query[i], doc[j]>
//
// A CPU fp32 reference is computed and the two are compared.
//
// Usage:
//   CHAMFER_CORE_KERNEL [config.json]
//   (defaults to ../config/config_kernel.json relative to the binary)

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

#include <vecflow_chamfer/kernels.hpp>

static inline void check_cuda(cudaError_t r, const char* expr, const char* file, int line) {
  if (r != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(r) << " at " << file << ":" << line
              << " (" << expr << ")\n";
    std::exit(EXIT_FAILURE);
  }
}
#define CHECK_CUDA(x) check_cuda((x), #x, __FILE__, __LINE__)

struct KernelConfig {
  int num_docs;
  int tokens_per_doc;
  int dim;
  int tokens_per_query;
  int warmup_runs;
  int num_runs;
  int sample_compare_count;
  bool verify_against_cpu;
};

static KernelConfig load_config(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Could not open config file: " + path);
  }
  nlohmann::json j;
  f >> j;

  KernelConfig c;
  c.num_docs              = j["data"]["num_docs"];
  c.tokens_per_doc        = j["data"]["tokens_per_doc"];
  c.dim                   = j["data"]["dim"];
  c.tokens_per_query      = j["data"]["tokens_per_query"];
  c.warmup_runs           = j["benchmark"]["warmup_runs"];
  c.num_runs              = j["benchmark"]["num_runs"];
  c.sample_compare_count  = j["output"].value("sample_compare_count", 10);
  c.verify_against_cpu    = j["output"].value("verify_against_cpu", true);
  return c;
}

static float cpu_chamfer_score(const float* query, const float* b,
                               int tokens_per_query, int n, int dim) {
  float sum_max = 0.0f;
  for (int i = 0; i < tokens_per_query; ++i) {
    const float* qrow = query + static_cast<size_t>(i) * dim;
    float row_max = -FLT_MAX;
    for (int j = 0; j < n; ++j) {
      const float* brow = b + static_cast<size_t>(j) * dim;
      float dot = 0.0f;
      for (int k = 0; k < dim; ++k) dot += qrow[k] * brow[k];
      row_max = std::max(row_max, dot);
    }
    sum_max += row_max;
  }
  return sum_max;
}

static std::string default_config_path(const char* relative) {
  try {
    auto exe = std::filesystem::canonical("/proc/self/exe");
    return (exe.parent_path() / relative).lexically_normal().string();
  } catch (...) {
    return relative;
  }
}

int main(int argc, char** argv) {
  const std::string config_path = (argc > 1)
    ? std::string(argv[1])
    : default_config_path("../config/config_kernel.json");
  std::cout << "[config] " << config_path << std::endl;

  KernelConfig cfg = load_config(config_path);

  std::cout << "  num_docs=" << cfg.num_docs
            << ", tokens_per_doc=" << cfg.tokens_per_doc
            << ", dim=" << cfg.dim
            << ", tokens_per_query=" << cfg.tokens_per_query
            << ", warmup_runs=" << cfg.warmup_runs
            << ", num_runs=" << cfg.num_runs << std::endl;

  std::vector<float>  h_query_f(static_cast<size_t>(cfg.tokens_per_query) * cfg.dim);
  std::vector<__half> h_query_h(h_query_f.size());

  const size_t total_b_rows = static_cast<size_t>(cfg.num_docs) * cfg.tokens_per_doc;
  std::vector<float>  h_dataset_f(total_b_rows * cfg.dim);
  std::vector<__half> h_dataset_h(h_dataset_f.size());

  std::vector<uint32_t> h_doc_offsets(static_cast<size_t>(cfg.num_docs) + 1);
  std::vector<uint32_t> h_doc_ids(cfg.num_docs);
  std::vector<float>    h_gpu_scores(cfg.num_docs, 0.0f);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < h_query_f.size(); ++i) {
    float v = dist(rng);
    h_query_f[i] = v;
    h_query_h[i] = __float2half(v);
  }
  for (size_t i = 0; i < h_dataset_f.size(); ++i) {
    float v = dist(rng);
    h_dataset_f[i] = v;
    h_dataset_h[i] = __float2half(v);
  }
  h_doc_offsets[0] = 0;
  for (int d = 0; d < cfg.num_docs; ++d) {
    h_doc_offsets[d + 1] = h_doc_offsets[d] + static_cast<uint32_t>(cfg.tokens_per_doc);
    h_doc_ids[d] = static_cast<uint32_t>(d);
  }

  __half*   d_query        = nullptr;
  __half*   d_dataset      = nullptr;
  uint32_t* d_doc_offsets  = nullptr;
  uint32_t* d_doc_ids      = nullptr;
  float*    d_scores       = nullptr;
  CHECK_CUDA(cudaMalloc(&d_query,       h_query_h.size()       * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_dataset,     h_dataset_h.size()     * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_doc_offsets, h_doc_offsets.size()   * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_doc_ids,     h_doc_ids.size()       * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_scores,      h_gpu_scores.size()    * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_query,       h_query_h.data(),     h_query_h.size()      * sizeof(__half),   cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_dataset,     h_dataset_h.data(),   h_dataset_h.size()    * sizeof(__half),   cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_doc_offsets, h_doc_offsets.data(), h_doc_offsets.size()  * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_doc_ids,     h_doc_ids.data(),     h_doc_ids.size()      * sizeof(uint32_t), cudaMemcpyHostToDevice));

  for (int i = 0; i < cfg.warmup_runs; ++i) {
    vecflow_chamfer::kernels::chamfer_score(
      cfg.num_docs, cfg.tokens_per_query, cfg.dim,
      d_query, d_dataset, d_doc_ids, d_doc_offsets, d_scores);
    CHECK_CUDA(cudaGetLastError());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  using clock = std::chrono::high_resolution_clock;
  using ms = std::chrono::duration<double, std::milli>;
  double total_ms = 0.0;
  for (int i = 0; i < cfg.num_runs; ++i) {
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t0 = clock::now();
    vecflow_chamfer::kernels::chamfer_score(
      cfg.num_docs, cfg.tokens_per_query, cfg.dim,
      d_query, d_dataset, d_doc_ids, d_doc_offsets, d_scores);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = clock::now();
    total_ms += ms(t1 - t0).count();
  }
  double avg_ms = total_ms / static_cast<double>(cfg.num_runs);

  std::cout << "\n[kernel] avg time: " << std::fixed << std::setprecision(4)
            << avg_ms << " ms for " << cfg.num_docs << " docs\n";
  std::cout << "[kernel] throughput: " << std::setprecision(2)
            << (cfg.num_docs * 1000.0 / avg_ms) << " docs/s\n";

  CHECK_CUDA(cudaMemcpy(h_gpu_scores.data(), d_scores,
                        h_gpu_scores.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  if (cfg.verify_against_cpu) {
    std::vector<float> h_cpu_scores(cfg.num_docs, 0.0f);
    auto cpu_t0 = clock::now();
    for (int d = 0; d < cfg.num_docs; ++d) {
      size_t b_offset_rows = static_cast<size_t>(h_doc_offsets[d]);
      const float* b_ptr = h_dataset_f.data() + b_offset_rows * cfg.dim;
      h_cpu_scores[d] = cpu_chamfer_score(h_query_f.data(), b_ptr,
                                          cfg.tokens_per_query, cfg.tokens_per_doc, cfg.dim);
    }
    auto cpu_t1 = clock::now();
    std::cout << "[cpu] verify time: " << ms(cpu_t1 - cpu_t0).count() << " ms\n";

    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    int max_diff_idx = -1;
    for (int d = 0; d < cfg.num_docs; ++d) {
      double diff = std::abs(static_cast<double>(h_gpu_scores[d]) -
                             static_cast<double>(h_cpu_scores[d]));
      if (diff > max_abs_diff) { max_abs_diff = diff; max_diff_idx = d; }
      mean_abs_diff += diff;
    }
    mean_abs_diff /= static_cast<double>(cfg.num_docs);
    std::cout << "[verify] mean_abs_diff=" << std::setprecision(6) << mean_abs_diff
              << ", max_abs_diff=" << max_abs_diff << "\n";

    int show = std::min(cfg.sample_compare_count, cfg.num_docs);
    std::cout << "\nSample comparisons (first " << show << "):\n";
    std::cout << "DocID  GPU          CPU          Diff\n";
    for (int i = 0; i < show; ++i) {
      double diff = static_cast<double>(h_gpu_scores[i]) -
                    static_cast<double>(h_cpu_scores[i]);
      std::cout << std::setw(5) << i << "  "
                << std::setw(12) << std::fixed << std::setprecision(6) << h_gpu_scores[i] << "  "
                << std::setw(12) << h_cpu_scores[i] << "  "
                << std::setw(12) << diff << "\n";
    }
    if (max_diff_idx >= 0) {
      std::cout << "\nWorst at DocID " << max_diff_idx
                << ": GPU=" << h_gpu_scores[max_diff_idx]
                << ", CPU=" << h_cpu_scores[max_diff_idx]
                << ", Diff=" << (h_gpu_scores[max_diff_idx] - h_cpu_scores[max_diff_idx])
                << "\n";
    }
  }

  CHECK_CUDA(cudaFree(d_query));
  CHECK_CUDA(cudaFree(d_dataset));
  CHECK_CUDA(cudaFree(d_doc_offsets));
  CHECK_CUDA(cudaFree(d_doc_ids));
  CHECK_CUDA(cudaFree(d_scores));
  return 0;
}
