#pragma once
#include "intrinsic.hh"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

/// @brief Perform warp-level reduction using AMD's DPP instructions.
/// @return The sum of all values in the warp stored in the lane `warpSize - 1`.
/// @acknowledgement: Borrowed from
/// https://medium.com/@hashem.hashemi/amds-dpp-operations-for-super-fast-reduction-29f9aa888376#
/// @note To get best performance, the input value should be a vector type
template <int VecSize = 1> __device__ void warp_reduce_sum_dpp(float (&val)[VecSize]) {
    constexpr int NOPS = VecSize > 2 ? 0 : (2 - VecSize);
    // Intra-row reduction
#pragma unroll
    for (int row_shr = 8; row_shr > 0; row_shr /= 2) {
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
            __v_add_f32_dpp_shr(val[i], val[i], val[i], row_shr); // ROW_SHR8
        }
        if constexpr (NOPS > 0)
            __nop<NOPS>();
    }
    // Inter-row reduction
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        __v_add_f32_dpp_bcast(val[i], val[i], val[i], 15); // BCAST15
    }
    if constexpr (NOPS > 0)
        __nop<NOPS>();
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
        __v_add_f32_dpp_bcast(val[i], val[i], val[i], 31); // BCAST31
    }
    if constexpr (NOPS > 0)
        __nop<NOPS>();
    __nop<1>();
}
