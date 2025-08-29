#include <3rdparty/argparse.hpp>
#include <lib/rocblas.hh>
#include <tools/helper.hh>
#include <tools/intrinsic.hh>
#include <vector>

float alpha = 2.0f;
std::vector<float> h_in;
std::vector<float> h_out;

float *d_alpha;
float *d_in;
float *d_out;

int N;
int timed_runs;
int warmup_runs;

constexpr int ThreadsPerBlock = 1024;

template <int ThreadsPerBlock, int VecSize = 1>
__global__ __launch_bounds__(ThreadsPerBlock) void kScale(const float *in, float *out, size_t size,
                                                          const float *alpha) {
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

template <int ThreadsPerBlock>
__global__ __launch_bounds__(ThreadsPerBlock) void kScale_buf(float *in, float *out, size_t size,
                                                              const float *alpha) {
    constexpr auto VecSize = 4;
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    BufferResource<float> buf(in, sizeof(float) * size);
    float r_in[VecSize], r_out[VecSize];
    auto tid                         = blockIdx.x * blockDim.x + threadIdx.x;
    *reinterpret_cast<vXf32 *>(r_in) = buf.load_x4(tid * sizeof(float) * VecSize, 0, 0);

    auto alpha_v = *alpha;

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
        r_out[j] = alpha_v * r_in[j];
    }
    buf.desc.base_addr_ = reinterpret_cast<uint64_t>(out);
    buf.store_x4(*reinterpret_cast<vXf32 *>(r_out), tid * sizeof(float) * VecSize, 0, 0);
}

template <int ThreadsPerBlock>
__global__ __launch_bounds__(ThreadsPerBlock) void pkScale_buf(float *in, float *out, size_t size,
                                                               const float *alpha) {
    constexpr auto VecSize = 4;
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    BufferResource<float> buf(in, sizeof(float) * size);
    BufferResource<float> obuf(out, sizeof(float) * size);

    auto bid = blockIdx.x;
    auto tid = threadIdx.x;

    auto numel   = divUp(size, gridDim.x);
    auto voffset = tid * VecSize * sizeof(float);

    float r_in[VecSize], r_out[VecSize];

    auto alpha_v = *alpha;

    for (int i = 0; i < numel; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<vXf32 *>(r_in) =
            buf.load_x4(voffset, (bid * numel + i) * sizeof(float), 0);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_out[j] = alpha_v * r_in[j];
        }
        obuf.store_x4(*reinterpret_cast<vXf32 *>(r_out), voffset, (bid * numel + i) * sizeof(float),
                      0);
    }
}

void sax(const float *in, float *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = alpha * in[i];
    }
}
int parse_args(int argc, char *argv[]) {
    argparse::ArgumentParser prog("ax");
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
    return 0;
}

#define REGISTER_SAX(name, vec_size)                                                               \
    profiler.add(                                                                                  \
        name,                                                                                      \
        [&]() {                                                                                    \
            constexpr auto VecSize = vec_size;                                                     \
            kScale<ThreadsPerBlock, VecSize>                                                       \
                <<<divUp(N, ThreadsPerBlock * VecSize), ThreadsPerBlock>>>(d_in, d_out, N,         \
                                                                           d_alpha);               \
        },                                                                                         \
        check_result, bytes, flops);

int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    h_in.resize(N);
    h_out.resize(N);
    Randomizer<float> rand(-1.0f, 1.0f);
    rand.fill_random(h_in.data(), h_in.size());

    sax(h_in.data(), h_out.data(), h_in.size());
    {
        check_runtime_api(hipMalloc(&d_in, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_out, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_alpha, sizeof(float)));
        check_runtime_api(hipMemcpy(d_alpha, &alpha, sizeof(float), H2D));
        check_runtime_api(hipMemcpy(d_in, h_in.data(), N * sizeof(float), H2D));
    }
    size_t bytes = 2 * N * sizeof(float);
    size_t flops = N;

    auto check_result = [&] { return validate_all(h_out.data(), d_out, h_out.size()); };
    auto reset_func   = [&] {
        check_runtime_api(hipMemcpy(d_in, h_in.data(), N * sizeof(float), H2D));
        check_runtime_api(hipMemset(d_out, 0, N * sizeof(float)));
    };

    Profiler profiler(timed_runs, warmup_runs, reset_func);

    RocBlasLv1<float> rocblas;
    profiler.add("rocBLAS", [&] { rocblas.scale(N, d_alpha, d_out, 1); }, nullptr, bytes, flops);

    REGISTER_SAX("ScaleV1", 1);
    REGISTER_SAX("ScaleV2", 2);
    REGISTER_SAX("ScaleV4", 4);
    profiler.add(
        "ScaleBuf",
        [&]() {
            kScale_buf<ThreadsPerBlock>
                <<<divUp(N, ThreadsPerBlock * 4), ThreadsPerBlock>>>(d_in, d_out, N, d_alpha);
        },
        check_result, bytes, flops);
    profiler.add(
        "PersistentScaleBuf",
        [&]() { pkScale_buf<ThreadsPerBlock><<<120, ThreadsPerBlock>>>(d_in, d_out, N, d_alpha); },
        check_result, bytes, flops);
    profiler.runAll();

    {
        check_runtime_api(hipFree(d_alpha));
        check_runtime_api(hipFree(d_in));
        check_runtime_api(hipFree(d_out));
    }

    return 0;
}