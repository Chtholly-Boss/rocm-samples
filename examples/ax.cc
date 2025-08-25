#include <tools/helper.hh>
#include <type_traits>
#include <vector>

float alpha = 2.0f;
std::vector<float> h_in;
std::vector<float> h_out;

float *d_in;
float *d_out;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

constexpr int ThreadsPerBlock = 256;
int NUM_SM                    = 0;

template <int ThreadsPerBlock, int VecSize = 1>
__global__ void kSax(const float *in, float *out, size_t size, float alpha) {
    using VT = typename std::conditional<
        VecSize == 4, float4, typename std::conditional<VecSize == 2, float2, float>::type>::type;
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * VecSize < size) {
        float r_in[VecSize], r_out[VecSize];
        *reinterpret_cast<VT *>(r_in) = *reinterpret_cast<const VT *>(in + tid * VecSize);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_out[j] = alpha * r_in[j];
        }
        *reinterpret_cast<VT *>(out + tid * VecSize) = *reinterpret_cast<const VT *>(r_out);
    }
}

template <int ThreadsPerBlock, int VecSize>
__global__ void pkSax(const float *in, float *out, size_t size, float alpha) {
    using VT = typename std::conditional<
        VecSize == 4, float4, typename std::conditional<VecSize == 2, float2, float>::type>::type;

    auto numel   = (size + gridDim.x - 1) / gridDim.x;
    auto b_start = blockIdx.x * numel;
    auto b_end   = min(b_start + numel, size);

    float r_in[VecSize];
    float r_out[VecSize];
    auto tid = threadIdx.x;
    for (size_t i = b_start + tid * VecSize; i < b_end; i += ThreadsPerBlock * VecSize) {
        *reinterpret_cast<VT *>(r_in) = *reinterpret_cast<const VT *>(in + i);
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
            r_out[j] = alpha * r_in[j];
        }
        *reinterpret_cast<VT *>(out + i) = *reinterpret_cast<const VT *>(r_out);
    }
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
int main(int argc, char *argv[]) {
    hipDeviceProp_t props;
    check_runtime_api(hipGetDeviceProperties(&props, 0));
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
    Profiler profiler(timed_runs, warmup_runs);

    auto check_result = [&]() {
        auto ret = validate(h_out.data(), d_out, h_out.size());
        check_runtime_api(hipMemset(d_out, 0, N * sizeof(float)));
        return ret;
    };

    profiler.add(
        "SaxVec1",
        [&]() {
            kSax<ThreadsPerBlock, 1>
                <<<divUp(N, ThreadsPerBlock), ThreadsPerBlock>>>(d_in, d_out, N, alpha);
        },
        check_result, bytes, flops);
    profiler.add(
        "SaxVec2",
        [&]() {
            kSax<ThreadsPerBlock, 2>
                <<<divUp(N, ThreadsPerBlock * 2), ThreadsPerBlock>>>(d_in, d_out, N, alpha);
        },
        check_result, bytes, flops);
    profiler.add(
        "SaxVec4",
        [&]() {
            kSax<ThreadsPerBlock, 4>
                <<<divUp(N, ThreadsPerBlock * 4), ThreadsPerBlock>>>(d_in, d_out, N, alpha);
        },
        check_result, bytes, flops);
    profiler.add(
        "pSaxVec1",
        [&]() {
            pkSax<ThreadsPerBlock, 1><<<NUM_SM, ThreadsPerBlock>>>(d_in, d_out, N, alpha);
        },
        check_result, bytes, flops);
    profiler.add(
        "pSaxVec2",
        [&]() {
            pkSax<ThreadsPerBlock, 2><<<NUM_SM, ThreadsPerBlock>>>(d_in, d_out, N, alpha);
        },
        check_result, bytes, flops);
    profiler.add(
        "pSaxVec4",
        [&]() {
            pkSax<ThreadsPerBlock, 4><<<NUM_SM, ThreadsPerBlock>>>(d_in, d_out, N, alpha);
        },
        check_result, bytes, flops);
    profiler.runAll();

    {
        check_runtime_api(hipFree(d_in));
        check_runtime_api(hipFree(d_out));
    }

    return 0;
}