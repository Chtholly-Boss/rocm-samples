#pragma once
#include <hip/hip_runtime.h>
// clang-format off
#define INTRISIC __device__ __forceinline__

typedef int   v4i32 __attribute__((ext_vector_type(4)));
typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef int   v1i32;

typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v2f32 __attribute__((ext_vector_type(2)));
typedef float v1f32;

/// @brief No operation (NOP) instruction
/// @tparam cycles Number of cycles to wait, valid range: 1~255
template <int cycles>
INTRISIC void __nop() { asm volatile("s_nop %[cyc]" : : [cyc] "i"(cycles)); }

/// ==============================================================================
/// Data Parallel Primitives (DPP)
/// ==============================================================================

INTRISIC void __v_add_f32_dpp_shr(float &dst, const float &src0, const float &src1, int row_shr,
                                                    int row_mask = 0xf, int bank_mask = 0xf, int bound_ctrl = 1) {
    asm volatile("v_add_f32_dpp %[dst], %[src0], %[src1], row_shr:%[row_shr], row_mask:%[row_mask], "
                 "bank_mask:%[bank_mask], bound_ctrl:%[bound_ctrl]"
                 : [dst] "=v"(dst)
                 : [src0] "v"(src0), [src1] "v"(src1), [row_shr] "i"(row_shr), [row_mask] "i"(row_mask),
                   [bank_mask] "i"(bank_mask), [bound_ctrl] "i"(bound_ctrl));
}

INTRISIC void __v_add_f32_dpp_bcast(float &dst, const float &src0, const float &src1, 
                                                      int row_bcast, int row_mask = 0xf, int bank_mask = 0xf, int bound_ctrl = 1) {
    asm volatile("v_add_f32_dpp %[dst], %[src0], %[src1], row_bcast:%[row_bcast], row_mask:%[row_mask], "
                 "bank_mask:%[bank_mask], bound_ctrl:%[bound_ctrl]"
                 : [dst] "=v"(dst)
                 : [src0] "v"(src0), [src1] "v"(src1), [row_bcast] "i"(row_bcast), [row_mask] "i"(row_mask),
                   [bank_mask] "i"(bank_mask), [bound_ctrl] "i"(bound_ctrl));
}

/// ==============================================================================
/// Vector Memory Buffer Instructions and Buffer Resource Descriptor
/// @see ISA Chapter [Vector Memory Operations: Vector Memory Buffer Instructions]
/// ==============================================================================

/// @see https://github.com/microsoft/llvm/blob/master/test/CodeGen/AMDGPU/llvm.amdgcn.raw.buffer.load.ll

INTRISIC v4i32 amdgcn_raw_buffer_load_v4i32(v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v4i32");

INTRISIC v2i32 amdgcn_raw_buffer_load_v2i32(v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v2i32");

INTRISIC v1i32 amdgcn_raw_buffer_load_v1i32(v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v1i32");

INTRISIC void amdgcn_raw_buffer_store_v4i32(v4i32 data, v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v4i32");

INTRISIC void amdgcn_raw_buffer_store_v2i32(v2i32 data, v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v2i32");

INTRISIC void amdgcn_raw_buffer_store_v1i32(v1i32 data, v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v1i32");

INTRISIC v4f32 amdgcn_raw_buffer_load_v4f32(v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v4f32");

INTRISIC v2f32 amdgcn_raw_buffer_load_v2f32(v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v2f32");

INTRISIC v1f32 amdgcn_raw_buffer_load_v1f32(v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.load.v1f32");

INTRISIC void amdgcn_raw_buffer_store_v4f32(v4f32 data, v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v4f32");

INTRISIC void amdgcn_raw_buffer_store_v2f32(v2f32 data, v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v2f32");

INTRISIC void amdgcn_raw_buffer_store_v1f32(v1f32 data, v4i32 rsrc, int voffset, int soffset, int aux) 
__asm("llvm.amdgcn.raw.buffer.store.v1f32");

/// @brief Select raw buffer load/store instruction based on data type and vector size
/// @tparam T        Data type, float or int
/// @tparam VecSize Vector size, 1, 2, or 4
template <typename T, int VecSize = 1>
struct RawBufferInstruction {
    typedef T VT __attribute__((ext_vector_type(VecSize)));
    INTRISIC VT load(v4i32 rsrc, int voffset, int soffset, int aux = 0) const {
        if constexpr (std::is_same_v<T, float>) {
            if constexpr (VecSize == 4)      return amdgcn_raw_buffer_load_v4f32(rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 2) return amdgcn_raw_buffer_load_v2f32(rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 1) return amdgcn_raw_buffer_load_v1f32(rsrc, voffset, soffset, aux);
        } else if constexpr (std::is_same_v<T, int>) {
            if constexpr (VecSize == 4)      return amdgcn_raw_buffer_load_v4i32(rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 2) return amdgcn_raw_buffer_load_v2i32(rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 1) return amdgcn_raw_buffer_load_v1i32(rsrc, voffset, soffset, aux);
        }
    }
    INTRISIC void store(VT data, v4i32 rsrc, int voffset, int soffset, int aux = 0) const {
        if constexpr (std::is_same_v<T, float>) {
            if constexpr (VecSize == 4)      amdgcn_raw_buffer_store_v4f32(data, rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 2) amdgcn_raw_buffer_store_v2f32(data, rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 1) amdgcn_raw_buffer_store_v1f32(data, rsrc, voffset, soffset, aux);
        } else if constexpr (std::is_same_v<T, int>) {
            if constexpr (VecSize == 4)      amdgcn_raw_buffer_store_v4i32(data, rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 2) amdgcn_raw_buffer_store_v2i32(data, rsrc, voffset, soffset, aux);
            else if constexpr (VecSize == 1) amdgcn_raw_buffer_store_v1i32(data, rsrc, voffset, soffset, aux);
        }
    }
};

/// @brief A wrapper of buffer resource for raw buffer load/store instructions
template <typename T>
union BufferResource {
    typedef T v1T;
    typedef T v2T __attribute__((ext_vector_type(2)));
    typedef T v4T __attribute__((ext_vector_type(4)));

    constexpr static int NUM_FORMAT = (std::is_same_v<T, float> ? 7 : (
                                       std::is_same_v<T, int> ? 5 :(
                                       std::is_same_v<T, uint> ? 4 : 0))
                                    ) << 12;
    constexpr static int DATA_FORMAT = 4 << 15; // 1 field with 4 bytes: U32/S32/F32

    v4i32 content;
    struct {
        uint64_t base_addr_;        // in bytes
        uint32_t range_;            // in bytes
        uint32_t other_;
    } desc;

    INTRISIC BufferResource(uint64_t base_addr, uint32_t range) {
        desc.base_addr_ = base_addr;
        desc.range_     = range;
        desc.other_     = NUM_FORMAT | DATA_FORMAT;
    }

    /// @brief Load a vector from the buffer
    /// @param voffset vector offset in bytes from VGPR
    /// @param soffset scalar offset in bytes from SGPR
    /// @param aux     auxiliary GLC/SLC control, GLC: 1, SLC: 2, default: 0
    INTRISIC v1T load_x1(int voffset, int soffset, int aux = 0) const {
        return RawBufferInstruction<T, 1>{}.load(content, voffset, soffset, aux);
    }
    INTRISIC v2T load_x2(int voffset, int soffset, int aux = 0) const {
        return RawBufferInstruction<T, 2>{}.load(content, voffset, soffset, aux);
    }
    INTRISIC v4T load_x4(int voffset, int soffset, int aux = 0) const {
        return RawBufferInstruction<T, 4>{}.load(content, voffset, soffset, aux);
    }

    /// @brief Store a vector to the buffer
    /// @param data    data to store
    /// @param voffset vector offset in bytes from VGPR
    /// @param soffset scalar offset in bytes from SGPR
    /// @param aux     auxiliary GLC/SLC control, GLC: 1, SLC: 2, default: 0
    INTRISIC void store_x1(v1T data, int voffset, int soffset, int aux) const {
        return RawBufferInstruction<T, 1>{}.store(data, content, voffset, soffset, aux);
    }
    INTRISIC void store_x2(v2T data, int voffset, int soffset, int aux) const {
        return RawBufferInstruction<T, 2>{}.store(data, content, voffset, soffset, aux);
    }
    INTRISIC void store_x4(v4T data, int voffset, int soffset, int aux) const {
        return RawBufferInstruction<T, 4>{}.store(data, content, voffset, soffset, aux);
    }
};

/// ==============================================================================
/// Sched Barrier
/// @see https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-intrinsics
/// ==============================================================================

struct SchedBarrier {
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
    /// @brief Allow specific instruction type to be scheduled after this barrier
    /// @param mask Bitmask of allowed instruction types to be scheduled after this barrier
    template <int mask> INTRISIC void allow() { __builtin_amdgcn_sched_barrier(mask); }
};
// clang-format on
