#include <3rdparty/argparse.hpp>
#include <lib/rocblas.hh>
#include <tools/helper.hh>
#include <tools/intrinsic.hh>
#include <tools/primitive.hh>
#include <vector>

std::vector<float> h_A;
std::vector<float> h_B;
float h_dot_out;

float *d_A;
float *d_B;
float *d_dot_out;
float *d_reduce_buf;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

constexpr int ThreadsPerBlock = 256;

template <int ThreadsPerBlock = 256>
__global__ void kDotIntraBlock(float *A, float *B, float *g_reduce_buf, int N) {
    constexpr auto NumWarps = ThreadsPerBlock / warpSize;
    constexpr auto VecSize  = 4;

    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));

    __shared__ float smem_reduce[NumWarps];

    BufferResource<float> buf_a(A, N * sizeof(float));
    BufferResource<float> buf_b(B, N * sizeof(float));

    auto bid    = blockIdx.x;
    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;

    constexpr auto master_lane = warpSize - 1;
    auto gid                   = bid * ThreadsPerBlock + tid;
    float sum[VecSize]         = {0.0f};
    float reg_a[VecSize];
    float reg_b[VecSize];
    if (gid * VecSize < N) {
        *reinterpret_cast<vXf32 *>(reg_a) = buf_a.load_x4(gid * VecSize * sizeof(float), 0);
        *reinterpret_cast<vXf32 *>(reg_b) = buf_b.load_x4(gid * VecSize * sizeof(float), 0);
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
            sum[i] += reg_a[i] * reg_b[i];
        }
    }
    warp_reduce_sum<false>(sum);
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

template <int ThreadsPerBlock = 256>
__global__ void kDotInterBlock(float *g_reduce_buf, float *out, int num_blocks) {
    constexpr auto NumWarps = ThreadsPerBlock / warpSize;
    constexpr auto VecSize  = 4;

    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));

    __shared__ float smem_reduce[NumWarps];

    BufferResource<float> buf(g_reduce_buf, num_blocks * sizeof(float));

    auto tid    = threadIdx.x;
    auto laneid = __lane_id();
    auto warpid = tid / warpSize;

    constexpr auto master_lane = warpSize - 1;
    float sum[VecSize]         = {0.0f};
    float reg_in[VecSize];
    for (int i = tid * VecSize; i < num_blocks; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<vXf32 *>(reg_in) = buf.load_x4(i * sizeof(float), 0);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            sum[j] += reg_in[j];
        }
    }
    warp_reduce_sum<false>(sum);
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

void dot_product(float *A, float *B, float *out, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += A[i] * B[i];
    }
    *out = sum;
}

void run_baseline() { dot_product(h_A.data(), h_B.data(), &h_dot_out, h_A.size()); }

int parse_args(int argc, char *argv[]) {
    argparse::ArgumentParser prog("memcpy");
    prog.add_argument("-n", "--size").help("vector size of A/B").required().scan<'i', int>();
    prog.add_argument("-t", "--timed_runs")
        .help("number of timed runs")
        .default_value(20)
        .scan<'i', int>();
    prog.add_argument("-w", "--warmup_runs")
        .help("number of warmup runs")
        .default_value(10)
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
    return 0;
}

int main(int argc, char *argv[]) {
    parse_args(argc, argv);

    h_A.resize(N);
    h_B.resize(N);
    Randomizer<float> rand(-1.0f, 1.0f);
    rand.fill_random(h_A.data(), h_A.size());
    rand.fill_random(h_B.data(), h_B.size());

    run_baseline();

    {
        check_runtime_api(hipMalloc(&d_A, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_B, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_dot_out, sizeof(float)));
        check_runtime_api(hipMalloc(&d_reduce_buf, N * sizeof(float)));
        check_runtime_api(hipMemcpy(d_A, h_A.data(), N * sizeof(float), H2D));
        check_runtime_api(hipMemcpy(d_B, h_B.data(), N * sizeof(float), H2D));
    }

    size_t bytes      = 2 * N * sizeof(float);
    size_t flops      = 2 * N;
    auto check_result = [&] { return validate_all(&h_dot_out, d_dot_out, 1, 0, 1e-2); };
    auto reset_func   = [&] { check_runtime_api(hipMemset(d_dot_out, 0, sizeof(float))); };
    RocBlasLv1<float> rocblas;
    Profiler profiler(timed_runs, warmup_runs, reset_func);
    profiler.add(
        "rocblas", [&]() { rocblas.dot(N, d_A, 1, d_B, 1, d_dot_out); }, check_result, bytes,
        flops);
    profiler.add(
        "Dot-2-phase",
        [&]() {
            constexpr auto VecSize = 4;
            auto num_blocks        = divUp(N, ThreadsPerBlock * VecSize);
            kDotIntraBlock<ThreadsPerBlock>
                <<<num_blocks, ThreadsPerBlock>>>(d_A, d_B, d_reduce_buf, N);
            kDotInterBlock<ThreadsPerBlock>
                <<<1, ThreadsPerBlock>>>(d_reduce_buf, d_dot_out, num_blocks);
        },
        check_result, bytes, flops);
    profiler.runAll();
    {
        check_runtime_api(hipFree(d_A));
        check_runtime_api(hipFree(d_B));
        check_runtime_api(hipFree(d_dot_out));
        check_runtime_api(hipFree(d_reduce_buf));
    }
}