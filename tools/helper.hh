#pragma once
#include <functional>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_cooperative_groups.h>
#include <iostream>
#include <random>
#include <vector>

#define check_runtime_api(call)                                                                    \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if (err != hipSuccess) {                                                                   \
            fprintf(stderr, "HIP error in file '%s' in line %i : %s.\n", __FILE__, __LINE__,       \
                    hipGetErrorString(err));                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__host__ __device__ __forceinline__ auto divUp(auto a, auto b) { return (a + b - 1) / b; }

/// @brief Validate the output data on device against the reference data on host using element-wise
/// comparison with given absolute and relative tolerances.
/// @param h_ref Pointer to the reference data on host.
/// @param d_out Pointer to the output data on device.
/// @param size Number of elements to validate.
/// @param atol Absolute tolerance.
/// @param rtol Relative tolerance.
/// @return True if the output data matches the reference data within the given tolerances, false
/// otherwise.
template <typename HT, typename DT>
bool validate_all(HT *h_ref, DT *d_out, size_t size, double atol = 1, double rtol = 1e-2) {
    auto h_out = (DT *)malloc(size * sizeof(DT));
    check_runtime_api(hipMemcpy(h_out, d_out, size * sizeof(DT), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < size; i++) {
        auto abs_err = std::abs(h_ref[i] - h_out[i]);
        auto rel_err = h_ref[i] == 0 ? 0.0 : abs_err / std::abs(h_ref[i]);
        if (atol > 0 and abs_err > atol) {
            fprintf(stderr, "Failed at [%zu]:", i);
            std::cerr << "ref = " << h_ref[i] << ", out = " << h_out[i] << ", abs_err = " << abs_err
                      << std::endl;
            free(h_out);
            return false;
        }
        if (rtol > 0 and rel_err > rtol) {
            fprintf(stderr, "Failed at [%zu]:", i);
            std::cerr << "ref = " << h_ref[i] << ", out = " << h_out[i] << ", rel_err = " << rel_err
                      << std::endl;
            free(h_out);
            return false;
        }
    }
    free(h_out);
    return true;
}

/// @brief Validate the output data on device against the reference data on host using RMSRE metric.
template <typename HT, typename DT>
bool validate_rmsre(HT *h_ref, DT *d_out, size_t size, double tol = 1e-2, double eps = 1e-12) {
    auto h_out = (DT *)malloc(size * sizeof(DT));
    check_runtime_api(hipMemcpy(h_out, d_out, size * sizeof(DT), hipMemcpyDeviceToHost));
    double rmsre = 0.0;
    for (size_t i = 0; i < size; i++) {
        double denom   = std::abs(h_ref[i]) + eps;
        double rel_err = (h_ref[i] - h_out[i]) / denom;
        rmsre += rel_err * rel_err;
    }
    rmsre = std::sqrt(rmsre / size);
    free(h_out);
    if (rmsre > tol) {
        std::cerr << "RMSRE " << rmsre << " exceeds tolerance " << tol << std::endl;
        return false;
    }
    return true;
}

class CPUTimer {
    using clock      = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;

  public:
    void start() { start_ = clock::now(); }
    void stop() { end_ = clock::now(); }
    float milliseconds() { return std::chrono::duration<float, std::milli>(end_ - start_).count(); }
    float seconds() { return milliseconds() / 1e3; }
    float microseconds() { return milliseconds() * 1e3; }

  private:
    time_point start_, end_;
};

template <typename Kernel, typename... Args>
void launchCooperativeKernel(Kernel kernel, dim3 gridDim, dim3 blockDim, int smem_size,
                             hipStream_t stream, Args... args) {
    void *kernelArgs[] = {(void *)&args...};
    check_runtime_api(hipLaunchCooperativeKernel((void *)(kernel), gridDim, blockDim, kernelArgs,
                                                 smem_size, stream));
}

class GPUTimer {
  public:
    GPUTimer() {
        check_runtime_api(hipEventCreate(&start_));
        check_runtime_api(hipEventCreate(&stop_));
    }
    ~GPUTimer() {
        check_runtime_api(hipEventDestroy(start_));
        check_runtime_api(hipEventDestroy(stop_));
    }
    void start() { check_runtime_api(hipEventRecord(start_, 0)); }
    void stop() {
        check_runtime_api(hipEventRecord(stop_, 0));
        check_runtime_api(hipEventSynchronize(stop_));
    }
    float milliseconds() {
        float ms;
        check_runtime_api(hipEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    float seconds() { return milliseconds() / 1e3; }
    float microseconds() { return milliseconds() * 1e3; }

  private:
    hipEvent_t start_, stop_;
};

class Profiler {
    struct Func {
        std::string name;
        std::function<void()> launch;
        std::function<bool()> validate;
        size_t bytes;
        size_t flops;
    };
    std::vector<Func> funcs;
    int timed_runs_;
    int warmup_runs_;
    GPUTimer timer_;
    size_t l2_size_;

  public:
    Profiler(int timed_runs = 10, int warmup_runs = 2)
        : timed_runs_(timed_runs), warmup_runs_(warmup_runs) {
        hipDeviceProp_t props;
        check_runtime_api(hipGetDeviceProperties(&props, 0));
        l2_size_ = props.l2CacheSize;
    }
    void add(const std::string &name, std::function<void()> launch,
             std::function<bool()> validate = nullptr, size_t bytes = 0, size_t flops = 0) {
        funcs.push_back({name, launch, validate, bytes, flops});
    }
    void runAll() {
        float *l2_flush;
        check_runtime_api(hipMalloc(&l2_flush, l2_size_));
        for (auto &f : funcs) {
            printf("name: %s\n", f.name.c_str());
            if (f.validate) {
                f.launch();
                check_runtime_api(hipDeviceSynchronize());
                if (f.validate()) {
                    printf("Validation: PASSED\n");
                }
            }
            for (int i = 0; i < warmup_runs_; i++) {
                f.launch();
            }
            check_runtime_api(hipDeviceSynchronize());
            float total_ms = 0;
            for (int i = 0; i < timed_runs_; i++) {
                check_runtime_api(hipMemset(l2_flush, 0, l2_size_)); // L2 cache flush
                timer_.start();
                f.launch();
                check_runtime_api(hipDeviceSynchronize());
                timer_.stop();
                total_ms += timer_.milliseconds();
            }
            auto avg_ms = total_ms / timed_runs_;
            printf("Average Time(%d runs): %.3f ms\n", timed_runs_, avg_ms);
            if (f.bytes > 0) {
                auto gbps = (float)f.bytes / 1e9 / (avg_ms / 1e3);
                printf("Throughput: %.3f GB/s\n", gbps);
            }
            if (f.flops > 0) {
                auto gflops = (float)f.flops / 1e9 / (avg_ms / 1e3);
                printf("Performance: %.3f GFLOPS\n", gflops);
            }
            std::cout << std::endl;
        }
        check_runtime_api(hipFree(l2_flush));
    }
};

template <typename T> class Randomizer {
  public:
    std::mt19937 gen;
    std::uniform_real_distribution<T> dist;
    Randomizer(T min, T max, unsigned int seed = 43) : gen(seed) {
        dist = std::uniform_real_distribution<T>(min, max);
    }
    void fill_random(T *data, size_t N) {
        for (size_t i = 0; i < N; i++) {
            data[i] = dist(gen);
        }
    }
    void fill_const(T *data, size_t N, T val) {
        for (size_t i = 0; i < N; i++) {
            data[i] = val;
        }
    }
    void fill_seq(T *data, size_t N, size_t mod = 0) {
        for (size_t i = 0; i < N; i++) {
            data[i] = mod == 0 ? (T)i : (T)(i % mod);
        }
    }
};

/// \brief Formats a range of elements to a pretty string.
/// \tparam BidirectionalIterator - must implement the BidirectionalIterator concept and
/// must be dereferencable in host code. Its value type must be formattable to
/// \p std::ostream.
template<class BidirectionalIterator>
inline std::string format_range(const BidirectionalIterator begin, const BidirectionalIterator end)
{
    std::stringstream sstream;
    sstream << "[ ";
    for(auto it = begin; it != end; ++it)
    {
        sstream << *it;
        if(it != std::prev(end))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}
