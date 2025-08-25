#include <tools/helper.hh>
#include <vector>

std::vector<float> h_data;
float *d_data;
float *d_data_dup;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

int main(int argc, char *argv[]) {
    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        timed_runs = atoi(argv[2]);
    if (argc > 3)
        warmup_runs = atoi(argv[3]);
    printf("N:%d timed_runs:%d warmup_runs:%d\n", N, timed_runs, warmup_runs);
    h_data.resize(N);
    {
        check_runtime_api(hipMalloc(&d_data, N * sizeof(float)));
        check_runtime_api(hipMalloc(&d_data_dup, N * sizeof(float)));
    }
    Profiler profiler(timed_runs, warmup_runs);
    size_t bytes = 2 * N * sizeof(float);
    profiler.add(
        "Memcpy H2D",
        [&]() { hipMemcpy(d_data, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice); },
        nullptr, bytes, 0);
    profiler.add(
        "Memcpy D2H",
        [&]() { hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost); },
        nullptr, bytes, 0);
    profiler.add(
        "Memcpy D2D",
        [&]() { hipMemcpy(d_data_dup, d_data, N * sizeof(float), hipMemcpyDeviceToDevice); },
        nullptr, bytes, 0);
    profiler.runAll();
    {
        check_runtime_api(hipFree(d_data));
        check_runtime_api(hipFree(d_data_dup));
    }
    return 0;
}