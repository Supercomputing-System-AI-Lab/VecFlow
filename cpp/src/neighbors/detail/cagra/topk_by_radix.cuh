/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "topk_for_cagra/topk_core.cuh"

namespace cuvs::neighbors::cagra::detail {
namespace filtered_single_cta_search {

template <unsigned MAX_INTERNAL_TOPK>
struct topk_by_radix_sort_base {
  static constexpr std::uint32_t smem_size        = MAX_INTERNAL_TOPK * 2 + 2048 + 8;
  static constexpr std::uint32_t state_bit_lenght = 0;
  static constexpr std::uint32_t vecLen           = 2;  // TODO
};
template <unsigned MAX_INTERNAL_TOPK, class IdxT, class = void>
struct topk_by_radix_sort : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {};

template <unsigned MAX_INTERNAL_TOPK, class IdxT>
struct topk_by_radix_sort<MAX_INTERNAL_TOPK, IdxT, std::enable_if_t<((MAX_INTERNAL_TOPK <= 64))>>
  : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {
  RAFT_DEVICE_INLINE_FUNCTION void operator()(uint32_t topk,
                                              uint32_t batch_size,
                                              uint32_t len_x,
                                              const uint32_t* _x,
                                              const IdxT* _in_vals,
                                              uint32_t* _y,
                                              IdxT* _out_vals,
                                              uint32_t* work,
                                              uint32_t* _hints,
                                              bool sort,
                                              uint32_t* _smem)
  {
    std::uint8_t* const state = reinterpret_cast<std::uint8_t*>(work);
    topk_cta_11_core<topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_lenght,
                     topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen,
                     64,
                     32,
                     IdxT>(topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
  }
};

#define TOP_FUNC_PARTIAL_SPECIALIZATION(V)                                           \
  template <unsigned MAX_INTERNAL_TOPK, class IdxT>                                  \
  struct topk_by_radix_sort<                                                         \
    MAX_INTERNAL_TOPK,                                                               \
    IdxT,                                                                            \
    std::enable_if_t<((MAX_INTERNAL_TOPK <= V) && (2 * MAX_INTERNAL_TOPK > V))>>     \
    : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {                                   \
    RAFT_DEVICE_INLINE_FUNCTION void operator()(uint32_t topk,                       \
                                                uint32_t batch_size,                 \
                                                uint32_t len_x,                      \
                                                const uint32_t* _x,                  \
                                                const IdxT* _in_vals,                \
                                                uint32_t* _y,                        \
                                                IdxT* _out_vals,                     \
                                                uint32_t* work,                      \
                                                uint32_t* _hints,                    \
                                                bool sort,                           \
                                                uint32_t* _smem)                     \
    {                                                                                \
      assert(blockDim.x >= V / 4);                                                   \
      std::uint8_t* state = (std::uint8_t*)work;                                     \
      topk_cta_11_core<topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_lenght, \
                       topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen,           \
                       V,                                                            \
                       V / 4,                                                        \
                       IdxT>(                                                        \
        topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);       \
    }                                                                                \
  };
TOP_FUNC_PARTIAL_SPECIALIZATION(128);
TOP_FUNC_PARTIAL_SPECIALIZATION(256);
TOP_FUNC_PARTIAL_SPECIALIZATION(512);
TOP_FUNC_PARTIAL_SPECIALIZATION(1024);
#undef TOP_FUNC_PARTIAL_SPECIALIZATION

}  // namespace filtered_single_cta_search
}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::cagra::detail {
namespace single_cta_search {

struct topk_by_radix_sort_base {
  static constexpr std::uint32_t state_bit_length = 0;
  static constexpr std::uint32_t vecLen           = 2;  // TODO

  static constexpr uint32_t smem_size(uint32_t max_itopk) { return max_itopk * 2 + 2048 + 8; }
};

RAFT_DEVICE_INLINE_FUNCTION void topk_cta_11_core_wrapper_256(uint32_t topk,
                                                              uint32_t len_x,
                                                              const uint32_t* _x,
                                                              const uint32_t* _in_vals,
                                                              uint32_t* _y,
                                                              uint32_t* _out_vals,
                                                              uint8_t* state,
                                                              uint32_t* _hints,
                                                              bool sort,
                                                              uint32_t* _smem)
{
  topk_cta_11_core<topk_by_radix_sort_base::state_bit_length,
                   topk_by_radix_sort_base::vecLen,
                   256,
                   64,
                   uint32_t>(topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
}

RAFT_DEVICE_INLINE_FUNCTION void topk_cta_11_core_wrapper_512(uint32_t topk,
                                                              uint32_t len_x,
                                                              const uint32_t* _x,
                                                              const uint32_t* _in_vals,
                                                              uint32_t* _y,
                                                              uint32_t* _out_vals,
                                                              uint8_t* state,
                                                              uint32_t* _hints,
                                                              bool sort,
                                                              uint32_t* _smem)
{
  topk_cta_11_core<topk_by_radix_sort_base::state_bit_length,
                   topk_by_radix_sort_base::vecLen,
                   512,
                   128,
                   uint32_t>(topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
}

template <class IdxT>
struct topk_by_radix_sort : topk_by_radix_sort_base {
  RAFT_DEVICE_INLINE_FUNCTION void operator()(uint32_t max_itopk,
                                              uint32_t topk,
                                              uint32_t len_x,
                                              const uint32_t* _x,
                                              const IdxT* _in_vals,
                                              uint32_t* _y,
                                              IdxT* _out_vals,
                                              uint32_t* work,
                                              uint32_t* _hints,
                                              bool sort,
                                              uint32_t* _smem)
  {
    if constexpr (std::is_same_v<IdxT,
                                 uint32_t>) {  // use a non-template wrapper function to avoid
                                               // pre-inlining the topk_cta_11_core function (vs
                                               // post-inlining, this impacts register pressure)
      std::uint8_t* const state = reinterpret_cast<std::uint8_t*>(work);
      if (max_itopk <= 256) {
        topk_cta_11_core_wrapper_256(
          topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
      } else {
        assert(max_itopk <= 512);
        topk_cta_11_core_wrapper_512(
          topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
      }
    } else {  // currently, unused
      std::uint8_t* const state = reinterpret_cast<std::uint8_t*>(work);
      if (max_itopk <= 256) {
        topk_cta_11_core<topk_by_radix_sort_base::state_bit_length,
                         topk_by_radix_sort_base::vecLen,
                         256,
                         64,
                         IdxT>(
          topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
      } else {
        assert(max_itopk <= 512);
        topk_cta_11_core<topk_by_radix_sort_base::state_bit_length,
                         topk_by_radix_sort_base::vecLen,
                         512,
                         128,
                         IdxT>(
          topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
      }
    }
  }
};

}  // namespace single_cta_search
}  // namespace cuvs::neighbors::cagra::detail
