#include <3rdparty/argparse.hpp>
#include <tools/helper.hh>
#include <vector>

std::vector<float> h_data;
float *d_data;
float *d_data_dup;

int N           = 65536;
int timed_runs  = 20;
int warmup_runs = 5;

int parse_args(int argc, char *argv[]) {
    argparse::ArgumentParser prog("memcpy");
    prog.add_argument("-n", "--size").help("number of dword(4 bytes)").required().scan<'i', int>();
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