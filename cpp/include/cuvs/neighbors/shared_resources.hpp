/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <type_traits>

namespace shared_resources {

inline thread_local int thread_id = 0;
inline thread_local int n_threads = 1;

struct non_blocking_stream {
  non_blocking_stream() { cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking); }
  ~non_blocking_stream() noexcept
  {
    if (stream_ != nullptr) { cudaStreamDestroy(stream_); }
  }
  non_blocking_stream(non_blocking_stream const&) = delete;
  non_blocking_stream(non_blocking_stream&& other) noexcept { std::swap(stream_, other.stream_); }
  auto operator=(non_blocking_stream const&) -> non_blocking_stream& = delete;
  auto operator=(non_blocking_stream&&) -> non_blocking_stream&      = delete;
  [[nodiscard]] auto view() const noexcept -> cudaStream_t { return stream_; }

 private:
  cudaStream_t stream_{nullptr};
};

namespace detail {
	inline std::vector<non_blocking_stream> global_stream_pool(0);
	inline std::mutex gsp_mutex;
}

/**
 * Get a stream associated with the current benchmark thread.
 *
 * Note, the streams are reused between the benchmark cases.
 * This makes it easier to profile and analyse multiple benchmark cases in one timeline using tools
 * like nsys.
 */
inline auto get_stream_from_global_pool() -> cudaStream_t
{
  std::lock_guard guard(detail::gsp_mutex);
  if (static_cast<int>(detail::global_stream_pool.size()) < n_threads) {
    detail::global_stream_pool.resize(n_threads);
  }
  return detail::global_stream_pool[thread_id].view();
}

/** Report a more verbose error with a backtrace when OOM occurs on RMM side. */
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
 * This container keeps the part of raft state that should be shared among multiple copies of raft
 * handles (in different CPU threads).
 * An example of this is an RMM memory resource: if we had an RMM memory pool per thread, we'd
 * quickly run out of memory.
 */
class shared_raft_resources {
 public:
  using pool_mr_type  = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  using mr_type       = rmm::mr::failure_callback_resource_adaptor<pool_mr_type>;
  using large_mr_type = rmm::mr::managed_memory_resource;

  shared_raft_resources()
  try : orig_resource_{rmm::mr::get_current_device_resource()},
    pool_resource_(orig_resource_, 1024 * 1024 * 1024ull),
    resource_(&pool_resource_, rmm_oom_callback, nullptr), large_mr_() {
    rmm::mr::set_current_device_resource(&resource_);
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

  auto get_large_memory_resource() noexcept
  {
    return static_cast<rmm::mr::device_memory_resource*>(&large_mr_);
  }

 private:
  rmm::mr::device_memory_resource* orig_resource_;
  pool_mr_type pool_resource_;
  mr_type resource_;
  large_mr_type large_mr_;
};

/**
 * This struct is used by multiple raft benchmark wrappers. It serves as a thread-safe keeper of
 * shared and private GPU resources (see below).
 *
 * - Accessing the same `configured_raft_resources` from concurrent threads is not safe.
 * - Accessing the copies of `configured_raft_resources` from concurrent threads is safe.
 * - There must be at most one "original" `configured_raft_resources` at any time, but as many
 *   copies of it as needed (modifies the program static state).
 */
class configured_raft_resources {
 public:
  /**
   * This constructor has the shared state passed unmodified but creates the local state anew.
   * It's used by the copy constructor.
   */
  explicit configured_raft_resources(const std::shared_ptr<shared_raft_resources>& shared_res)
    : shared_res_{shared_res},
      res_{std::make_unique<raft::device_resources>(
        rmm::cuda_stream_view(get_stream_from_global_pool()))}
  {
    // set the large workspace resource to the raft handle, but without the deleter
    // (this resource is managed by the shared_res).
    raft::resource::set_large_workspace_resource(
      *res_,
      std::shared_ptr<rmm::mr::device_memory_resource>(shared_res_->get_large_memory_resource(),
                                                       raft::void_op{}));
  }

  /** Default constructor creates all resources anew. */
  configured_raft_resources() : configured_raft_resources{std::make_shared<shared_raft_resources>()}
  {
  }

  configured_raft_resources(configured_raft_resources&&);
  auto operator=(configured_raft_resources&&) -> configured_raft_resources&;
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

  /** Get the main stream */
  [[nodiscard]] auto get_sync_stream() const noexcept
  {
    return raft::resource::get_cuda_stream(*res_);
  }

 private:
  /** The resources shared among multiple raft handles / threads. */
  std::shared_ptr<shared_raft_resources> shared_res_;
  /**
   * Until we make the use of copies of raft::resources thread-safe, each benchmark wrapper must
   * have its own copy of it.
   */
  std::unique_ptr<raft::device_resources> res_ = std::make_unique<raft::device_resources>();
};



};