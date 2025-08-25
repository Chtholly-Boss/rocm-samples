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

template <bool shfl, int VecSize = 1>
__device__ __forceinline__ void warp_reduce_sum(float (&val)[VecSize]) {
    if constexpr (shfl) {
        warp_reduce_sum_shfl(val);
    } else {
        warp_reduce_sum_dpp(val);
    }
}

template <bool shfl, int ThreadsPerBlock = 256, int VecSize = 1>
__global__ void kReduceIntraBlock(float *in, float *out, int N, float *g_reduce_buf) {
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    constexpr auto NumWarps = ThreadsPerBlock / warpSize;
    __shared__ float smem_reduce[NumWarps];

    auto bid    = blockIdx.x;
    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;

    constexpr auto master_lane = shfl ? 0 : (warpSize - 1);

    auto gid           = bid * ThreadsPerBlock + tid;
    float sum[VecSize] = {0.0f};
    if (gid * VecSize < N) {
        *reinterpret_cast<vXf32 *>(sum) = *reinterpret_cast<vXf32 *>(in + gid * VecSize);
    }
    warp_reduce_sum<shfl>(sum);
    if (laneid == master_lane) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
            acc += sum[i];
        }
        smem_reduce[warpid] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < NumWarps; i++) {
            sum += smem_reduce[i];
        }
        g_reduce_buf[bid] = sum;
    }
}

template <bool shfl, int ThreadsPerBlock = 256, int VecSize = 1>
__global__ void kReduceInterBlock(float *g_reduce_buf, float *out, int num_blocks) {
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    constexpr auto NumWarps = ThreadsPerBlock / warpSize;

    __shared__ float smem_reduce[NumWarps];

    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;

    constexpr auto master_lane = shfl ? 0 : (warpSize - 1);
    float sum[VecSize]         = {0.0f};
    float reg_in[VecSize];
    for (int i = tid * VecSize; i < num_blocks; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<vXf32 *>(reg_in) = *reinterpret_cast<vXf32 *>(g_reduce_buf + i);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            sum[j] += reg_in[j];
        }
    }
    warp_reduce_sum<shfl>(sum);
    if (laneid == master_lane) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
            acc += sum[i];
        }
        smem_reduce[warpid] = acc;
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

template <int VecSize = 1, bool shfl = true, typename T>
void launch_reduce(T *d_in, T *d_out, int N, T *d_reduce_buf) {
    auto num_blocks = divUp(N, ThreadsPerBlock * VecSize);
    kReduceIntraBlock<shfl, ThreadsPerBlock, VecSize>
        <<<num_blocks, ThreadsPerBlock>>>(d_in, d_out, N, d_reduce_buf);
    kReduceInterBlock<shfl, ThreadsPerBlock, VecSize>
        <<<1, ThreadsPerBlock>>>(d_reduce_buf, d_out, num_blocks);
}

int main(int argc, char *argv[]) {
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
        check_runtime_api(hipMalloc(&d_reduce_buf, 1024 * sizeof(float)));
    }

    check_runtime_api(hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice));
    auto check_result = [&]() {
        auto ret = validate(&h_out, d_out, 1);
        check_runtime_api(hipMemset(d_out, 0, sizeof(float)));
        return ret;
    };

    size_t bytes = N * sizeof(float);
    size_t flops = N;
    Profiler profiler(timed_runs, warmup_runs);
    profiler.add(
        "Reduce_shfl", [&]() { launch_reduce<1, true>(d_in, d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_shfl_vec2", [&]() { launch_reduce<2, true>(d_in, d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_shfl_vec4", [&]() { launch_reduce<4, true>(d_in, d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_dpp", [&]() { launch_reduce<1, false>(d_in, d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_dpp_vec2", [&]() { launch_reduce<2, false>(d_in, d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "Reduce_dpp_vec4", [&]() { launch_reduce<4, false>(d_in, d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.runAll();

    {
        check_runtime_api(hipFree(d_in));
        check_runtime_api(hipFree(d_out));
        check_runtime_api(hipFree(d_reduce_buf));
    }

    return 0;
}
