/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

namespace shared_resources {

inline thread_local int thread_id = 0;
inline thread_local int n_threads = 1;

struct non_blocking_stream {
  non_blocking_stream() { cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking); }
  ~non_blocking_stream() noexcept
  {
    if (stream_ != nullptr) { cudaStreamDestroy(stream_); }
  }
  non_blocking_stream(non_blocking_stream const&)                    = delete;
  auto operator=(non_blocking_stream const&) -> non_blocking_stream& = delete;
  non_blocking_stream(non_blocking_stream&& other) noexcept { std::swap(stream_, other.stream_); }
  auto operator=(non_blocking_stream&&) -> non_blocking_stream& = default;
  [[nodiscard]] auto view() const noexcept -> cudaStream_t { return stream_; }

 private:
  cudaStream_t stream_{nullptr};
};

namespace detail {
inline std::vector<non_blocking_stream> global_stream_pool(0);
inline std::mutex gsp_mutex;
}  // namespace detail

/**
 * Per-thread stream from a shared pool. Reused across runs within the same thread so tools
 * like nsys can group operations across benchmark cases.
 */
inline auto get_stream_from_global_pool() -> cudaStream_t
{
  std::lock_guard<std::mutex> guard(detail::gsp_mutex);
  if (static_cast<int>(detail::global_stream_pool.size()) < n_threads) {
    detail::global_stream_pool.resize(n_threads);
  }
  return detail::global_stream_pool[thread_id].view();
}

/** Verbose error reporter for RMM OOM. */
inline auto rmm_oom_callback(std::size_t bytes, void*) -> bool
{
  auto cuda_status = cudaGetLastError();
  size_t free      = 0;
  size_t total     = 0;
  RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
  RAFT_FAIL(
    "Failed to allocate %zu bytes using RMM memory resource. "
    "NB: latest cuda status = %s, free memory = %zu, total memory = %zu.",
    bytes,
    cudaGetErrorName(cuda_status),
    free,
    total);
}

/**
 * Holds raft state shared across multiple raft handles (typically across CPU threads).
 * Owns a 1 GiB pool wrapped with an OOM-reporting failure callback, and a managed-memory
 * resource for large/staging workspaces.
 *
 * Uses the rmm 26.06+ owning-`any_resource` API: `set_current_device_resource` takes/returns
 * `cuda::mr::any_resource<cuda::mr::device_accessible>`, and `pool_memory_resource` /
 * `failure_callback_resource_adaptor<>` are both constructed from `any_resource` upstreams.
 */
class shared_raft_resources {
 public:
  using large_mr_type = rmm::mr::managed_memory_resource;

  shared_raft_resources()
  try : large_mr_() {
    // Wrap the previous device resource with a 1 GiB pool, then with an OOM-reporting
    // failure-callback adaptor, and install as the current device resource.
    orig_resource_ =
      rmm::mr::set_current_device_resource(rmm::mr::failure_callback_resource_adaptor<>{
        rmm::mr::pool_memory_resource{rmm::mr::get_current_device_resource_ref(),
                                      1024 * 1024 * 1024ull},
        rmm_oom_callback,
        nullptr});
  } catch (const std::exception& e) {
    auto cuda_status = cudaGetLastError();
    size_t free      = 0;
    size_t total     = 0;
    RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
    RAFT_FAIL(
      "Failed to initialize shared raft resources (NB: latest cuda status = %s, free memory = %zu, "
      "total memory = %zu): %s",
      cudaGetErrorName(cuda_status),
      free,
      total,
      e.what());
  }

  shared_raft_resources(shared_raft_resources&&)                               = delete;
  auto operator=(shared_raft_resources&&) -> shared_raft_resources&            = delete;
  shared_raft_resources(const shared_raft_resources& res)                      = delete;
  auto operator=(const shared_raft_resources& other) -> shared_raft_resources& = delete;

  ~shared_raft_resources() noexcept { rmm::mr::set_current_device_resource(orig_resource_); }

  /** Returns the large/managed memory resource as a CCCL ref. */
  auto get_large_memory_resource() noexcept -> rmm::device_async_resource_ref { return large_mr_; }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> orig_resource_;
  large_mr_type large_mr_;
};

/**
 * Per-thread raft resources sharing an underlying memory pool with other copies.
 *
 * Accessing the same `configured_raft_resources` from concurrent threads is unsafe.
 * Accessing copies (one per thread) is safe.
 */
class configured_raft_resources {
 public:
  /** Build a wrapper sharing the supplied `shared_raft_resources` state. */
  explicit configured_raft_resources(const std::shared_ptr<shared_raft_resources>& shared_res)
    : shared_res_{shared_res},
      res_{std::make_unique<raft::device_resources>(
        rmm::cuda_stream_view(get_stream_from_global_pool()))}
  {
    raft::resource::set_large_workspace_resource(
      *res_, raft::mr::device_resource{shared_res_->get_large_memory_resource()});
  }

  /** Default constructor builds fresh shared and per-thread resources. */
  configured_raft_resources() : configured_raft_resources{std::make_shared<shared_raft_resources>()}
  {
  }

  configured_raft_resources(configured_raft_resources&&) noexcept = default;
  auto operator=(configured_raft_resources&&) noexcept -> configured_raft_resources& = default;
  ~configured_raft_resources() = default;

  configured_raft_resources(const configured_raft_resources& res)
    : configured_raft_resources{res.shared_res_}
  {
  }
  auto operator=(const configured_raft_resources& other) -> configured_raft_resources&
  {
    this->shared_res_ = other.shared_res_;
    return *this;
  }

  operator raft::resources&() noexcept { return *res_; }              // NOLINT
  operator const raft::resources&() const noexcept { return *res_; }  // NOLINT

  /** Get the main stream backing this wrapper. */
  [[nodiscard]] auto get_sync_stream() const noexcept
  {
    return raft::resource::get_cuda_stream(*res_);
  }

 private:
  std::shared_ptr<shared_raft_resources> shared_res_;
  std::unique_ptr<raft::device_resources> res_ = std::make_unique<raft::device_resources>();
};

}  // namespace shared_resources
