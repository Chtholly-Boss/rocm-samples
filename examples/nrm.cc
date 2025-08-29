#include <3rdparty/argparse.hpp>
#include <lib/rocblas.hh>
#include <tools/helper.hh>
#include <tools/intrinsic.hh>
#include <tools/primitive.hh>
#include <vector>

std::vector<float> h_data;
std::vector<float> h_ref;
float *d_out;
float *d_reduce_buf;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

constexpr int ThreadsPerBlock = 256;
int NUM_SM                    = 0;

template <typename T> void normalize(T *x, size_t n) {
    T nrm2 = 0;
    for (size_t i = 0; i < n; i++) {
        nrm2 += x[i] * x[i];
    }
    nrm2 = std::sqrt(nrm2);
    for (size_t i = 0; i < n; i++) {
        x[i] /= nrm2;
    }
}

/// @brief Persistent kernel to perform Reduce + Axpy for normalization
template <int ThreadsPerBlock = 256>
__global__ __launch_bounds__(ThreadsPerBlock) void pkNormalize(float *v, int N,
                                                               float *g_reduce_buf) {
    namespace cg               = cooperative_groups;
    constexpr auto NumWarps    = ThreadsPerBlock / warpSize;
    constexpr auto master_lane = warpSize - 1;
    constexpr auto VecSize     = 4;
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));

    __shared__ float smem_reduce[NumWarps];
    __shared__ float smem_v[12 * 1024];

    auto grid    = cg::this_grid();
    auto bid     = blockIdx.x;
    auto tid     = threadIdx.x;
    auto warp_id = tid / warpSize;
    auto lane_id = __lane_id();

    auto numel = divUp(N, gridDim.x);

    float sum[VecSize] = {0.0f};
    float r_v[VecSize];

    BufferResource<float> buf_v(v, N * sizeof(float));
    auto voffset = tid * VecSize;
    auto start   = bid * numel;
    /// Step 0: load data to shared memory and compute local square sum
    for (int i = 0; i < numel; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<vXf32 *>(r_v) =
            buf_v.load_x4(voffset * sizeof(float), (start + i) * sizeof(float), 0);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            sum[j] += r_v[j] * r_v[j]; // square for nrm
        }
        *reinterpret_cast<vXf32 *>(smem_v + i + voffset) = *reinterpret_cast<vXf32 *>(r_v);
    }
    warp_reduce_sum_dpp(sum);
    if (lane_id == master_lane) {
        float acc = 0.0f;
        for (int i = 0; i < VecSize; i++) {
            acc += sum[i]; // square sum for nrm2
        }
        smem_reduce[warp_id] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < NumWarps; i++) {
            acc += smem_reduce[i];
        }
        g_reduce_buf[bid] = acc;
    }

    grid.sync();

    __shared__ float s_nrm2;
    float val[1];
    val[0] = tid < gridDim.x ? g_reduce_buf[tid] : 0.0f;
    warp_reduce_sum_dpp(val);
    if (lane_id == master_lane) {
        smem_reduce[warp_id] = val[0];
    }
    __syncthreads();
    if (tid == 0) {
        float square_nrm2 = 0.0f;
        for (int i = 0; i < NumWarps; i++) {
            square_nrm2 += smem_reduce[i];
        }
        s_nrm2 = sqrt(square_nrm2);
    }
    __syncthreads();
    // Step 2: element-wise normalization
    float alpha = 1.0f / s_nrm2;
    for (int i = 0; i < numel; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<vXf32 *>(r_v) = *reinterpret_cast<vXf32 *>(smem_v + i + voffset);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_v[j] *= alpha;
        }
        buf_v.store_x4(*reinterpret_cast<vXf32 *>(r_v), voffset * sizeof(float),
                       (start + i) * sizeof(float), 0);
    }
}

template <bool shfl, int ThreadsPerBlock = 256, int VecSize = 1>
__global__ __launch_bounds__(ThreadsPerBlock) void kReduceIntraBlock(float *in, float *g_reduce_buf,
                                                                     int N) {
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
__global__ __launch_bounds__(ThreadsPerBlock) void kReduceInterBlock(float *g_reduce_buf,
                                                                     float *out, int num_blocks) {
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
        *out = sqrt(sum); //  1/nrm2 for normalization
    }
}

template <int ThreadsPerBlock>
__global__ __launch_bounds__(ThreadsPerBlock) void kSax(float *in, size_t size, float *alpha) {
    constexpr auto VecSize = 4;
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));

    BufferResource<float> buf_in(in, size * sizeof(float));
    auto tid     = blockIdx.x * ThreadsPerBlock + threadIdx.x;
    auto alpha_v = 1.0f / *alpha;
    float r_in[VecSize], r_out[VecSize];
    *reinterpret_cast<vXf32 *>(r_in) = buf_in.load_x4(tid * VecSize * sizeof(float), 0, 0);
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
        r_out[j] = alpha_v * r_in[j];
    }
    buf_in.store_x4(*reinterpret_cast<vXf32 *>(r_out), tid * VecSize * sizeof(float), 0, 0);
}

template <bool shfl, int VecSize> void launch_nrm2_multi_phase(float *v, int N, float *reduce_buf) {
    int num_blocks = divUp(N, ThreadsPerBlock * VecSize);
    kReduceIntraBlock<shfl, ThreadsPerBlock, VecSize>
        <<<num_blocks, ThreadsPerBlock>>>(const_cast<float *>(v), reduce_buf, N);
    kReduceInterBlock<shfl, ThreadsPerBlock, VecSize>
        <<<1, ThreadsPerBlock>>>(reduce_buf, reduce_buf, num_blocks); // write to buf[0]
    kSax<ThreadsPerBlock>
        <<<divUp(N, ThreadsPerBlock * VecSize), ThreadsPerBlock>>>(v, N, reduce_buf);
}

int parse_args(int argc, char *argv[]) {
    argparse::ArgumentParser prog("nrm");
    prog.add_argument("-n", "--size").help("vector size").required().scan<'i', int>();
    prog.add_argument("-t", "--timed_runs")
        .help("number of timed runs")
        .default_value(20)
        .scan<'i', int>();
    prog.add_argument("-w", "--warmup_runs")
        .help("number of warmup runs")
        .default_value(10)
        .scan<'i', int>();
    prog.add_argument("-d", "--device")
        .help("which device to use")
        .default_value(0)
        .scan<'i', int>();
    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << prog;
        exit(1);
    }
    N           = prog.get<int>("--size");
    timed_runs  = prog.get<int>("--timed_runs");
    warmup_runs = prog.get<int>("--warmup_runs");

    hipDeviceProp_t props;
    check_runtime_api(hipGetDeviceProperties(&props, prog.get<int>("--device")));
    NUM_SM = props.multiProcessorCount;
    return 0;
}

int main(int argc, char *argv[]) {
    parse_args(argc, argv);

    h_data.resize(N);
    h_ref.resize(N);
    Randomizer<float> rand(-1.0f, 1.0f);
    rand.fill_random(h_data.data(), h_data.size());
    memcpy(h_ref.data(), h_data.data(), N * sizeof(float));
    normalize(h_ref.data(), h_ref.size());

    {
        check_runtime_api(hipMalloc(&d_out, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_reduce_buf, N * sizeof(float)));
        check_runtime_api(
            hipMemcpy(d_out, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice));
    }

    size_t bytes      = 2 * N * sizeof(float);
    size_t flops      = 3 * N; // 2 * N (mul, add) + N (div)
    auto check_result = [&] { return validate_all(h_ref.data(), d_out, h_ref.size()); };
    auto reset_func   = [&] {
        check_runtime_api(
            hipMemcpy(d_out, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice));
    };
    Profiler profiler(timed_runs, warmup_runs, reset_func);

    profiler.add(
        "pkNormalize",
        [=]() {
            launchCooperativeKernel((pkNormalize<ThreadsPerBlock>), dim3(NUM_SM),
                                    dim3(ThreadsPerBlock), 0, 0, d_out, N, d_reduce_buf);
        },
        check_result, bytes, flops);
    profiler.add(
        "kNrm2-3phase-Vec-1", [&] { launch_nrm2_multi_phase<true, 1>(d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "kNrm2-3phase-Vec-2", [&] { launch_nrm2_multi_phase<true, 2>(d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    profiler.add(
        "kNrm2-3phase-Vec-4", [&] { launch_nrm2_multi_phase<true, 4>(d_out, N, d_reduce_buf); },
        check_result, bytes, flops);
    RocBlasLv1<float> rocblas;
    profiler.add(
        "rocblas",
        [&]() {
            constexpr auto VecSize = 4;
            float alpha;
            rocblas.nrm2(N, d_out, 1, &alpha);
            alpha = 1.0f / alpha;
            rocblas.scale(N, &alpha, d_out, 1);
        },
        check_result, bytes, flops);
    profiler.runAll();
    {
        check_runtime_api(hipFree(d_out));
        check_runtime_api(hipFree(d_reduce_buf));
    }

    return 0;
}