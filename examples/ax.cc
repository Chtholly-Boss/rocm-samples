#include <3rdparty/argparse.hpp>
#include <lib/rocblas.hh>
#include <tools/helper.hh>
#include <tools/intrinsic.hh>
#include <vector>

float alpha = 2.0f;
std::vector<float> h_in;
std::vector<float> h_out;

float *d_in;
float *d_out;

int N;
int timed_runs;
int warmup_runs;

constexpr int ThreadsPerBlock = 256;
int NUM_SM                    = 0;

template <int ThreadsPerBlock, int VecSize = 1>
__global__ void kSax(const float *in, float *out, size_t size, float alpha) {
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * VecSize < size) {
        float r_in[VecSize], r_out[VecSize];
        *reinterpret_cast<vXf32 *>(r_in) = *reinterpret_cast<const vXf32 *>(in + tid * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_out[j] = alpha * r_in[j];
        }
        *reinterpret_cast<vXf32 *>(out + tid * VecSize) = *reinterpret_cast<const vXf32 *>(r_out);
    }
}

template <int ThreadsPerBlock, int VecSize>
__global__ void pkSax(const float *in, float *out, size_t size, float alpha) {
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));

    auto numel   = (size + gridDim.x - 1) / gridDim.x;
    auto b_start = blockIdx.x * numel;
    auto b_end   = min(b_start + numel, size);

    float r_in[VecSize];
    float r_out[VecSize];
    auto tid = threadIdx.x;
    for (size_t i = b_start + tid * VecSize; i < b_end; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<vXf32 *>(r_in) = *reinterpret_cast<const vXf32 *>(in + i);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_out[j] = alpha * r_in[j];
        }
        *reinterpret_cast<vXf32 *>(out + i) = *reinterpret_cast<const vXf32 *>(r_out);
    }
}

template <int ThreadsPerBlock, int VecSize = 1>
__global__ void kSax_buf(float *in, float *out, size_t size, float alpha) {
    typedef float vXf32 __attribute__((ext_vector_type(VecSize)));
    BufferResource<float, VecSize> buf(reinterpret_cast<uint64_t>(in), sizeof(float) * size);
    BufferResource<float, VecSize> buf_out(reinterpret_cast<uint64_t>(out), sizeof(float) * size);
    float r_in[VecSize], r_out[VecSize];
    auto tid                         = blockIdx.x * blockDim.x + threadIdx.x;
    *reinterpret_cast<vXf32 *>(r_in) = buf.load(tid * sizeof(float) * VecSize, 0, 0);
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
        r_out[j] = alpha * r_in[j];
    }
    buf_out.store(*reinterpret_cast<vXf32 *>(r_out), tid * sizeof(float) * VecSize, 0, 0);
}

void sax(const float *in, float *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = alpha * in[i];
    }
}

void run_baseline() {
    CPUTimer timer;
    assert(h_out.size() == h_in.size());
    sax(h_in.data(), h_out.data(), h_in.size());
    for (int i = 0; i < warmup_runs; i++) {
        sax(h_in.data(), h_out.data(), h_in.size());
    }
    float ms = 0;
    for (int i = 0; i < timed_runs; i++) {
        timer.start();
        sax(h_in.data(), h_out.data(), h_in.size());
        timer.stop();
        ms += timer.milliseconds();
    }
    printf("Baseline: %f ms\n", ms / timed_runs);
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

    hipDeviceProp_t props;
    check_runtime_api(hipGetDeviceProperties(&props, prog.get<int>("--device")));
    NUM_SM = props.multiProcessorCount;
    return 0;
}

#define REGISTER_SAX(name, vec_size)                                                               \
    profiler.add(                                                                                  \
        name,                                                                                      \
        [&]() {                                                                                    \
            constexpr auto VecSize = vec_size;                                                     \
            kSax<ThreadsPerBlock, VecSize>                                                         \
                <<<divUp(N, ThreadsPerBlock * VecSize), ThreadsPerBlock>>>(d_in, d_out, N, alpha); \
        },                                                                                         \
        check_result, bytes, flops);

#define REGISTER_PSAX(name, vec_size)                                                              \
    profiler.add(                                                                                  \
        name,                                                                                      \
        [&]() {                                                                                    \
            constexpr auto VecSize = vec_size;                                                     \
            pkSax<ThreadsPerBlock, VecSize><<<NUM_SM, ThreadsPerBlock>>>(d_in, d_out, N, alpha);   \
        },                                                                                         \
        check_result, bytes, flops);

#define REGISTER_SAX_BUF(name, vec_size)                                                           \
    profiler.add(                                                                                  \
        name,                                                                                      \
        [&]() {                                                                                    \
            constexpr auto VecSize = vec_size;                                                     \
            kSax_buf<ThreadsPerBlock, VecSize>                                                     \
                <<<divUp(N, ThreadsPerBlock * VecSize), ThreadsPerBlock>>>(d_in, d_out, N, alpha); \
        },                                                                                         \
        check_result, bytes, flops);

int main(int argc, char *argv[]) {
    parse_args(argc, argv);
    h_in.resize(N);
    h_out.resize(N);
    Randomizer<float> rand(-1.0f, 1.0f);
    rand.fill_random(h_in.data(), h_in.size());

    run_baseline();
    {
        check_runtime_api(hipMalloc(&d_in, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_out, N * sizeof(float)));
        check_runtime_api(hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice));
    }
    size_t bytes = 2 * N * sizeof(float);
    size_t flops = N;
    Profiler profiler(timed_runs, warmup_runs,
                      [&] { check_runtime_api(hipMemset(d_out, 0, N * sizeof(float))); });

    auto check_result = [&] { return validate_all(h_out.data(), d_out, h_out.size()); };

    REGISTER_SAX("SaxVec1", 1);
    REGISTER_SAX("SaxVec2", 2);
    REGISTER_SAX("SaxVec4", 4);
    REGISTER_PSAX("PSaxVec1", 1);
    REGISTER_PSAX("PSaxVec2", 2);
    REGISTER_PSAX("PSaxVec4", 4);
    REGISTER_SAX_BUF("SaxBufVec2", 2);
    REGISTER_SAX_BUF("SaxBufVec4", 4);
    RocBlasLv1<float> rocblas;
    profiler.add("rocBLAS", [&] { rocblas.scale(N, &alpha, d_out, 1); }, nullptr, bytes, flops);
    profiler.runAll();

    {
        check_runtime_api(hipFree(d_in));
        check_runtime_api(hipFree(d_out));
    }

    return 0;
}