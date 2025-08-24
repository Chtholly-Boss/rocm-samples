#include "tools/primitive.hh"
#include <hip/hip_cooperative_groups.h>
#include <tools/helper.hh>
#include <type_traits>
#include <vector>

std::vector<float> h_in;
float h_out;
float *d_in;
float *d_out;
float *d_reduce_buf;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

constexpr int ThreadsPerBlock = 256;
int NUM_SM                    = 0;

__global__ void kReduce_shfl(float *in, float *out, int size, float *g_reduce_buf) {}

template <typename T> __device__ __forceinline__ T warp_reduce_sum_shfl(T val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <bool shfl, int ThreadsPerBlock = 256>
__global__ void kReduce(float *in, float *out, int N, float *g_reduce_buf) {
    namespace cg = cooperative_groups;

    constexpr auto NumWarps    = ThreadsPerBlock / warpSize;
    constexpr auto master_lane = shfl ? 0 : (warpSize - 1);
    __shared__ float smem_reduce[NumWarps];

    auto bid    = blockIdx.x;
    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;
    auto grid   = cg::this_grid();
    /// 1st Phase: Intra-block reduction
    auto numel   = (N + gridDim.x - 1) / gridDim.x;
    auto b_start = bid * numel;
    auto b_end   = min(b_start + numel, (unsigned)N);

    float sum[1] = {0.0f};
    for (int i = b_start + tid; i < b_end; i += ThreadsPerBlock) {
        sum[0] += in[i];
    }
    if constexpr (shfl) {
        sum[0] = warp_reduce_sum_shfl(sum[0]);
    } else {
        warp_reduce_sum_dpp(sum);
    }
    if (laneid == master_lane) {
        smem_reduce[warpid] = sum[0];
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < NumWarps; i++) {
            sum += smem_reduce[i];
        }
        g_reduce_buf[bid] = sum;
    }

    grid.sync();

    /// 2nd Phase: Inter-block reduction
    if (bid == 0) {
        sum[0] = (tid < gridDim.x) ? g_reduce_buf[tid] : 0.0f;
        if constexpr (shfl) {
            sum[0] = warp_reduce_sum_shfl(sum[0]);
        } else {
            warp_reduce_sum_dpp(sum);
        }
        if (laneid == master_lane) {
            smem_reduce[warpid] = sum[0];
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < NumWarps; i++) {
                sum += smem_reduce[i];
            }
            *out = sum;
        }
    }
}

template <int ThreadsPerBlock = 256, int VecSize = 1>
__global__ void kReduce_vdpp(float *in, float *out, int N, float *g_reduce_buf) {
    namespace cg = cooperative_groups;
    using VT     = typename std::conditional<
            VecSize == 4, float4, typename std::conditional<VecSize == 2, float2, float>::type>::type;
    constexpr auto NumWarps    = ThreadsPerBlock / warpSize;
    constexpr auto master_lane = warpSize - 1;

    __shared__ float smem_reduce[NumWarps];

    auto bid    = blockIdx.x;
    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;
    auto grid   = cg::this_grid();

    float acc[VecSize] = {0.0f};
    /// 1st Phase: Intra-block reduction
    auto numel   = (N + gridDim.x - 1) / gridDim.x;
    auto b_start = bid * numel;
    auto b_end   = min(b_start + numel, (unsigned)N);

    float r_in[VecSize];
    for (int i = b_start + tid * VecSize; i < b_end; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<VT *>(r_in) = *reinterpret_cast<const VT *>(&in[i]);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            acc[v] += r_in[v];
        }
    }
    warp_reduce_sum_dpp(acc);
    if (laneid == master_lane) {
        float sum = 0.0f;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            sum += acc[v];
        }
        smem_reduce[warpid] = sum;
    }

    grid.sync();

    /// 2nd Phase: Inter-block reduction
    if (bid == 0) {
        float sum[1] = {0.0f};
        sum[0]       = (tid < gridDim.x) ? g_reduce_buf[tid] : 0.0f;
        warp_reduce_sum_dpp(sum);
        if (laneid == master_lane) {
            smem_reduce[warpid] = sum[0];
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < NumWarps; i++) {
                sum += smem_reduce[i];
            }
            *out = sum;
        }
    }
}

template <int ThreadsPerBlock = 256, int VecGroups = 1>
__global__ void kReduce_vdpp_ext(float *in, float *out, int N, float *g_reduce_buf) {
    namespace cg               = cooperative_groups;
    constexpr int VecSize      = 4;
    using VT                   = float4;
    constexpr auto NumWarps    = ThreadsPerBlock / warpSize;
    constexpr auto master_lane = warpSize - 1;

    __shared__ float smem_reduce[NumWarps];

    auto bid    = blockIdx.x;
    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;
    auto grid   = cg::this_grid();

    float acc[VecSize] = {0.0f};
    /// 1st Phase: Intra-block reduction
    auto numel   = (N + gridDim.x - 1) / gridDim.x;
    auto b_start = bid * numel;
    auto b_end   = min(b_start + numel, (unsigned)N);

    float r_in[VecGroups][VecSize];
    for (int i = b_start + tid * VecGroups * VecSize; i < b_end;
         i += ThreadsPerBlock * VecGroups * VecSize) {
#pragma unroll
        for (int g = 0; g < VecGroups; g++) {
            *reinterpret_cast<VT *>(r_in[g]) = *reinterpret_cast<const VT *>(&in[i + g * VecSize]);
        }
#pragma unroll
        for (int g = 0; g < VecGroups; g++) {
#pragma unroll
            for (int v = 0; v < VecSize; v++) {
                acc[v] += r_in[g][v];
            }
        }
    }
    warp_reduce_sum_dpp(acc);
    if (laneid == master_lane) {
        float sum = 0.0f;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            sum += acc[v];
        }
        smem_reduce[warpid] = sum;
    }

    grid.sync();

    /// 2nd Phase: Inter-block reduction
    if (bid == 0) {
        float sum[1] = {0.0f};
        sum[0]       = (tid < gridDim.x) ? g_reduce_buf[tid] : 0.0f;
        warp_reduce_sum_dpp(sum);
        if (laneid == master_lane) {
            smem_reduce[warpid] = sum[0];
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < NumWarps; i++) {
                sum += smem_reduce[i];
            }
            *out = sum;
        }
    }
}

float reduce(const float *in, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += in[i];
    }
    return sum;
}

void run_baseline() {
    CPUTimer timer;
    h_out = reduce(h_in.data(), h_in.size());
    for (int i = 0; i < warmup_runs; i++) {
        h_out = reduce(h_in.data(), h_in.size());
    }
    float ms = 0;
    for (int i = 0; i < timed_runs; i++) {
        timer.start();
        h_out = reduce(h_in.data(), h_in.size());
        timer.stop();
        ms += timer.milliseconds();
    }
    printf("Baseline: %f ms\n", ms / timed_runs);
}

int main(int argc, char *argv[]) {
    hipDeviceProp_t props;
    check_runtime_api(hipGetDeviceProperties(&props, 0));
    printf("Device: %s\n", props.name);
    NUM_SM = props.multiProcessorCount;
    printf("NUM_SM=%d\n", NUM_SM);

    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        timed_runs = atoi(argv[2]);
    if (argc > 3)
        warmup_runs = atoi(argv[3]);
    printf("N:%d timed_runs:%d warmup_runs:%d\n", N, timed_runs, warmup_runs);

    h_in.resize(N);
    Randomizer<float> rand(-1.0f, 1.0f);
    rand.fill_random(h_in.data(), h_in.size());
    run_baseline();

    {
        check_runtime_api(hipMalloc(&d_in, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_out, sizeof(float)));
        check_runtime_api(hipMalloc(&d_reduce_buf, NUM_SM * sizeof(float)));
    }

    check_runtime_api(hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice));
    auto check_result = [&]() { return validate(&h_out, d_out, 1); };

    size_t bytes = N * sizeof(float);
    size_t flops = N;
    Profiler profiler(timed_runs, warmup_runs);
    profiler.add(
        "Reduce_shfl",
        [&]() {
            launchCooperativeKernel(kReduce<true>, NUM_SM, ThreadsPerBlock, 0, 0, d_in, d_out, N,
                                    d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_dpp",
        [&]() {
            launchCooperativeKernel(kReduce<false>, NUM_SM, ThreadsPerBlock, 0, 0, d_in, d_out, N,
                                    d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_vdpp_2",
        [&]() {
            launchCooperativeKernel(kReduce_vdpp<ThreadsPerBlock, 2>, NUM_SM, ThreadsPerBlock, 0, 0,
                                    d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_vdpp_4",
        [&]() {
            launchCooperativeKernel(kReduce_vdpp<ThreadsPerBlock, 4>, NUM_SM, ThreadsPerBlock, 0, 0,
                                    d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_vdpp_ext_2",
        [&]() {
            launchCooperativeKernel(kReduce_vdpp_ext<ThreadsPerBlock, 2>, NUM_SM, ThreadsPerBlock,
                                    0, 0, d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_vdpp_ext_4",
        [&]() {
            launchCooperativeKernel(kReduce_vdpp_ext<ThreadsPerBlock, 4>, NUM_SM, ThreadsPerBlock,
                                    0, 0, d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_vdpp_ext_8",
        [&]() {
            launchCooperativeKernel(kReduce_vdpp_ext<ThreadsPerBlock, 8>, NUM_SM, ThreadsPerBlock,
                                    0, 0, d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_vdpp_ext_12",
        [&]() {
            launchCooperativeKernel(kReduce_vdpp_ext<ThreadsPerBlock, 12>, NUM_SM, ThreadsPerBlock,
                                    0, 0, d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.runAll();

    {
        check_runtime_api(hipFree(d_in));
        check_runtime_api(hipFree(d_out));
        check_runtime_api(hipFree(d_reduce_buf));
    }

    return 0;
}
