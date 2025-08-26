#include "tools/primitive.hh"
#include <hip/hip_cooperative_groups.h>
#include <tools/helper.hh>
#include <type_traits>
#include <vector>

std::vector<float> h_in;
std::vector<float> h_out;
float *d_in;
float *d_out;
float *d_reduce_buf;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

constexpr int ThreadsPerBlock = 256;
int NUM_SM                    = 0;

template <typename T> void normalize(T *x, T *y, size_t n) {
    T nrm2 = 0;
    for (size_t i = 0; i < n; i++) {
        nrm2 += x[i] * x[i];
    }
    nrm2 = std::sqrt(nrm2);
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] / nrm2;
    }
}

template <int GroupSize, typename T> __device__ __forceinline__ void warp_reduce_sum(T &val) {
#pragma unroll
    for (int offset = GroupSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
}

/// @brief Persistent kernel to perform Reduce + Axpy for normalization
template <int ThreadsPerBlock = 256>
__global__ void pkNormalize(const float *in, float *out, int N, float *g_reduce_buf) {
    namespace cg            = cooperative_groups;
    constexpr auto NumWarps = ThreadsPerBlock / warpSize;

    __shared__ float smem_reduce[NumWarps];

    auto grid    = cg::this_grid();
    auto tid     = threadIdx.x;
    auto bid     = blockIdx.x;
    auto lane_id = __lane_id();
    auto warp_id = tid / warpSize;

    auto numel = divUp(N, gridDim.x);
    auto start = bid * numel;
    auto end   = min(start + numel, N);

    /// Step 1: calculate nrm2
    float sum = 0.0f;
    for (int i = start + tid; i < end; i += ThreadsPerBlock) {
        sum += in[i] * in[i];
    }
    warp_reduce_sum<warpSize>(sum);
    if (lane_id == 0) {
        smem_reduce[warp_id] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        sum = 0.0f;
        for (int i = 0; i < NumWarps; i++) {
            sum += smem_reduce[i];
        }
        g_reduce_buf[bid] = sum;
    }

    grid.sync();

    __shared__ float s_nrm2;
    float square_nrm2 = 0.0f;
    float val         = tid < gridDim.x ? g_reduce_buf[tid] : 0.0f;
    warp_reduce_sum<NumWarps>(val);
    if (lane_id == 0) {
        smem_reduce[warp_id] = val;
    }
    __syncthreads();
    if (tid == 0) {
        square_nrm2 = 0.0f;
        for (int i = 0; i < NumWarps; i++) {
            square_nrm2 += smem_reduce[i];
        }
        s_nrm2 = sqrt(square_nrm2);
    }
    __syncthreads();
    // Step 2: element-wise normalization
    float alpha = 1.0f / s_nrm2;
    for (int i = start + tid; i < end; i += ThreadsPerBlock) {
        out[i] = in[i] * alpha;
    }
}

template <bool shfl, int ThreadsPerBlock = 256, int VecSize = 1>
__global__ void kReduceIntraBlock(float *in, float *g_reduce_buf, int N) {
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
    for (int i = 0; i < VecSize; i++) {
        sum[i] = sum[i] * sum[i]; // square for nrm2
    }
    warp_reduce_sum<shfl>(sum);
    if (laneid == master_lane) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
            acc += sum[i]; // square sum for nrm2
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
        *out = 1.0f / sqrt(sum); //  1/nrm2 for normalization
    }
}

template <int ThreadsPerBlock, int VecSize = 1>
__global__ void kSax(const float *in, float *out, size_t size, float *alpha) {
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    auto tid     = blockIdx.x * blockDim.x + threadIdx.x;
    auto alpha_v = *alpha;
    if (tid * VecSize < size) {
        float r_in[VecSize], r_out[VecSize];
        *reinterpret_cast<vXf32 *>(r_in) = *reinterpret_cast<const vXf32 *>(in + tid * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_out[j] = alpha_v * r_in[j];
        }
        *reinterpret_cast<vXf32 *>(out + tid * VecSize) = *reinterpret_cast<const vXf32 *>(r_out);
    }
}

void run_baseline() {
    CPUTimer timer;
    normalize(h_in.data(), h_out.data(), h_in.size());
    for (int i = 0; i < warmup_runs; i++) {
        normalize(h_in.data(), h_out.data(), h_in.size());
    }
    float ms = 0;
    for (int i = 0; i < timed_runs; i++) {
        timer.start();
        normalize(h_in.data(), h_out.data(), h_in.size());
        timer.stop();
        ms += timer.milliseconds();
    }
    printf("Baseline: %f ms\n", ms / timed_runs);
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    hipDeviceProp_t props;
    check_runtime_api(hipGetDeviceProperties(&props, 0));
    NUM_SM = props.multiProcessorCount;
    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        timed_runs = atoi(argv[2]);
    if (argc > 3)
        warmup_runs = atoi(argv[3]);
    printf("N:%d timed_runs:%d warmup_runs:%d\n", N, timed_runs, warmup_runs);

    h_in.resize(N);
    h_out.resize(N);
    Randomizer<float> rand(-1.0f, 1.0f);
    rand.fill_random(h_in.data(), h_in.size());
    run_baseline();

    {
        check_runtime_api(hipMalloc(&d_in, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_out, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_reduce_buf, N * sizeof(float)));
        check_runtime_api(hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice));
    }

    size_t bytes = 2 * N * sizeof(float);
    size_t flops = 3 * N; // 2 * N (mul, add) + N (div)
    Profiler profiler(timed_runs, warmup_runs);
    auto check_result = [&]() {
        auto ret = validate(h_out.data(), d_out, h_out.size());
        check_runtime_api(hipMemset(d_out, 0, N * sizeof(float)));
        return ret;
    };
    profiler.add(
        "pkNormalize",
        [=]() {
            launchCooperativeKernel((pkNormalize<ThreadsPerBlock>), dim3(NUM_SM),
                                    dim3(ThreadsPerBlock), 0, 0, d_in, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "kReduce + kSax",
        [=]() {
            constexpr auto VecSize = 4;
            int num_blocks         = divUp(N, ThreadsPerBlock * VecSize);
            kReduceIntraBlock<true, ThreadsPerBlock, VecSize>
                <<<num_blocks, ThreadsPerBlock>>>(d_in, d_reduce_buf, N);
            kReduceInterBlock<true, ThreadsPerBlock, VecSize>
                <<<1, ThreadsPerBlock>>>(d_reduce_buf, d_reduce_buf, num_blocks); // write to buf[0]
            kSax<ThreadsPerBlock, VecSize>
                <<<num_blocks, ThreadsPerBlock>>>(d_in, d_out, N, d_reduce_buf);
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