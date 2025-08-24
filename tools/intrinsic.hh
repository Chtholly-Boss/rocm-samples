#pragma once
#include <hip/hip_runtime.h>
// clang-format off
template <int cycles>
__device__ __forceinline__ void __nop() { asm volatile("s_nop %[cyc]" : : [cyc] "i"(cycles)); }

__device__ int4 amdgcn_raw_buffer_load_v4i32(int4 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ void amdgcn_raw_buffer_store_v4i32(int4 data, int4 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ float4 amdgcn_raw_buffer_load_v4f32(int4 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v4f32");

__device__ void amdgcn_raw_buffer_store_v4f32(float4 data, int4 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v4f32");

/// See ISA Chapter [Vector Memory Operations: Vector Memory Buffer Instructions]
union BufferResource {
    static constexpr unsigned kDataFormatU32Config = 4 << 15;
    enum { kNone = 0, kGLCBit = 1 << 0, kSLCBit = 1 << 1 };

    int4 content;
    struct {
        uintptr_t ptr;
        unsigned range;
        unsigned config;
    } v;

    __device__ __forceinline__ uint4 load(int voffset, int soffset, int aux) const {
        int4 v = amdgcn_raw_buffer_load_v4i32(content, voffset, soffset, aux);
        return *reinterpret_cast<const uint4 *>(&v);
    }

    __device__ __forceinline__ void store(int voffset, int soffset, int aux, uint4 data) const {
        int4 v = *reinterpret_cast<const int4 *>(&data);
        amdgcn_raw_buffer_store_v4i32(v, content, voffset, soffset, aux);
    }
};

__device__ __forceinline__ void __v_add_f32_dpp_shr(float &dst, const float &src0, const float &src1, int row_shr,
                                                    int row_mask = 0xf, int bank_mask = 0xf, int bound_ctrl = 1) {
    asm volatile("v_add_f32_dpp %[dst], %[src0], %[src1], row_shr:%[row_shr], row_mask:%[row_mask], "
                 "bank_mask:%[bank_mask], bound_ctrl:%[bound_ctrl]"
                 : [dst] "=v"(dst)
                 : [src0] "v"(src0), [src1] "v"(src1), [row_shr] "i"(row_shr), [row_mask] "i"(row_mask),
                   [bank_mask] "i"(bank_mask), [bound_ctrl] "i"(bound_ctrl));
}

__device__ __forceinline__ void __v_add_f32_dpp_bcast(float &dst, const float &src0, const float &src1, 
                                                      int row_bcast, int row_mask = 0xf, int bank_mask = 0xf, int bound_ctrl = 1) {
    asm volatile("v_add_f32_dpp %[dst], %[src0], %[src1], row_bcast:%[row_bcast], row_mask:%[row_mask], "
                 "bank_mask:%[bank_mask], bound_ctrl:%[bound_ctrl]"
                 : [dst] "=v"(dst)
                 : [src0] "v"(src0), [src1] "v"(src1), [row_bcast] "i"(row_bcast), [row_mask] "i"(row_mask),
                   [bank_mask] "i"(bank_mask), [bound_ctrl] "i"(bound_ctrl));
}

struct SchedBarrier {
    // see https://llvm.org/docs/AMDGPUUsage.html : llvm.amdgcn.sched.barrier
    enum {
        NONE           = 0x0,
        ALU            = 0x1,
        VALU           = 0x2,
        SALU           = 0x4,
        MFMA_WMMA      = 0x8,
        VMEM           = 0x10,
        VMEM_READ      = 0x20,
        VMEM_WRITE     = 0x40,
        DS             = 0x80,
        DS_READ        = 0x100,
        DS_WRITE       = 0x200,
        TRANSCENDENTAL = 0x400,
    };
    template <int mask> __device__ __forceinline__ void allow() { __builtin_amdgcn_sched_barrier(mask); }
};
// clang-format on
