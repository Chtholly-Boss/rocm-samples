#include <fstream>
#include <lib/rocsparse.hh>
#include <tools/helper.hh>
#include <tools/primitive.hh>
#include <vector>

/// @class Compressed Sparse Row Matrix
/// @tparam VT Value Type
/// @details
///     - rows: row pointer array of size (nrows + 1)
///     - cols: column index array of size (nnz)
///     - vals: value array of size (nnz)
/// @note Should be loaded from binary file
template <typename VT> class CsrSpM {
  public:
    uint nrows;
    uint ncols;
    uint nnz;
    uint *rows;
    uint *cols;
    VT *vals;

    CsrSpM() : nrows(0), ncols(0), nnz(0), rows(NULL), cols(NULL), vals(NULL) {};
    CsrSpM(uint nrows, uint ncols, uint nnz) : nrows(nrows), ncols(ncols), nnz(nnz) {
        rows = (uint *)malloc((nrows + 1) * sizeof(uint));
        memset(rows, 0, (nrows + 1) * sizeof(uint));
        cols = (uint *)malloc((nnz) * sizeof(uint));
        memset(cols, 0, (nnz) * sizeof(uint));
        vals = (VT *)malloc((nnz) * sizeof(VT));
        memset(vals, 0, (nnz) * sizeof(VT));
    };
    CsrSpM(const CsrSpM<VT> &A) {
        nrows = A.nrows;
        ncols = A.ncols;
        nnz   = A.nnz;
        rows  = (uint *)malloc((nrows + 1) * sizeof(uint));
        cols  = (uint *)malloc((nnz) * sizeof(uint));
        vals  = (VT *)malloc((nnz) * sizeof(VT));
        for (unsigned int i = 0; i < nrows + 1; i++) {
            rows[i] = A.rows[i];
        }
        for (unsigned int p = 0; p < nnz; p++) {
            vals[p] = A.vals[p];
            cols[p] = A.cols[p];
        }
    }
    void loadFromBinary(std::string mtx_path) {
        std::ifstream binFile(mtx_path, std::ios::in | std::ios::binary);

        if (!binFile.is_open()) {
            std::cerr << "can not open output file!" << std::endl;
            return;
        }
        binFile.read(reinterpret_cast<char *>(&nrows), sizeof(nrows));
        binFile.read(reinterpret_cast<char *>(&ncols), sizeof(ncols));
        binFile.read(reinterpret_cast<char *>(&nnz), sizeof(nnz));

        rows = (uint *)malloc((nrows + 1) * sizeof(uint));
        cols = (uint *)malloc(nnz * sizeof(uint));
        vals = (double *)malloc(nnz * sizeof(double));

        binFile.read(reinterpret_cast<char *>(rows), (nrows + 1) * sizeof(uint));
        binFile.read(reinterpret_cast<char *>(cols), nnz * sizeof(uint));
        binFile.read(reinterpret_cast<char *>(vals), nnz * sizeof(double));
        binFile.close();
        printf("nrows: %d, ncols: %d, nnz: %d\n", nrows, ncols, nnz);
    }
    ~CsrSpM() {
        if (rows)
            free(rows);
        if (cols)
            free(cols);
        if (vals)
            free(vals);
    }
};

CsrSpM<double> mtx;
std::vector<double> x;
std::vector<double> ref_y;

uint *d_rows;
uint *d_cols;
int *d_rows_i32;
int *d_cols_i32;
double *d_x, *d_y;
double *d_vals;
float *d_x_f, *d_y_f;
float *d_vals_f;

int timed_runs  = 20;
int warmup_runs = 5;

constexpr auto ThreadsPerBlock = 256u;
int NUM_SM                     = 0;

/// @brief Convert precision from SrcT to DstT
/// @details Launched with ceil_div(N, ThreadsPerBlock) blocks
/// @note only implicit conversion is supported
template <typename DstT, typename SrcT>
__global__ void kCvtPrecision(DstT *dst, SrcT *src, uint N) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dst[i] = src[i];
    }
}

/// @brief Compute Y = values * X[colInd] element-wisely
/// @details Launched with ceil_div(nnz, ThreadsPerBlock) blocks
/// @note This kernel serves as an upper bound of SpMV performance
template <int ThreadsPerBlock = 256, int VecSize = 1, typename T>
__global__ void kValuesDotX(uint *colInd, T *values, T *X, T *Y, uint nnz) {
    typedef T vXT __attribute__((ext_vector_type(VecSize)));
    typedef uint vXu32 __attribute__((ext_vector_type(VecSize)));

    auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    T res[VecSize];
    T r_vals[VecSize];
    uint r_colInd[VecSize];
    if (gid * VecSize < nnz) {
        *reinterpret_cast<vXT *>(r_vals)     = *reinterpret_cast<vXT *>(&values[gid * VecSize]);
        *reinterpret_cast<vXu32 *>(r_colInd) = *reinterpret_cast<vXu32 *>(&colInd[gid * VecSize]);
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
            res[i] = r_vals[i] * X[r_colInd[i]];
        }
    }
    /// prevent compiler optimizing away the whole kernel
    if (threadIdx.x == 0) {
        Y[0] = res[0];
    }
}

/// @brief Reduce partial sums by row pointer
/// @note This kernel is the bottleneck of the two-kernel SpMV approach
template <int ThreadsPerBlock = 256, typename T>
__global__ void kReduceByRowPtr(uint *rowPtr, T *in, T *out, uint N) {}

/// @brief Vectorized SpMV kernel with @param GroupSize threads per row
/// @details Launched with ceil_div(num_rows, ThreadsPerBlock / GroupSize) blocks
template <int ThreadsPerBlock = 256, int GroupSize = 4, typename T>
__global__ void kVectorSubXSpMV(uint *rowPtr, uint *colInd, T *values, T *X, T *Y, uint N) {
    constexpr auto NumGroups = ThreadsPerBlock / GroupSize;
    auto tid                 = threadIdx.x;
    auto group_id            = tid / GroupSize;
    auto lane_id             = tid % GroupSize;
    auto gid                 = blockIdx.x * NumGroups + group_id;
    if (gid >= N)
        return;
    T sum = 0.0;
    for (uint j = rowPtr[gid] + lane_id; j < rowPtr[gid + 1]; j += GroupSize) {
        sum += values[j] * X[colInd[j]];
    }
#pragma unroll
    for (int offset = GroupSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset);
    }
    if (lane_id == 0) {
        Y[gid] = sum;
    }
}

/// @brief Vectorized SpMV kernel with @param GroupSize threads per row
/// @details Launched with NUM_SM blocks, each block processes ceil_div(num_rows, NUM_SM) rows
template <int ThreadsPerBlock = 256, int GroupSize = 4, typename T>
__global__ void pkVectorSubXSpMV(uint *rowPtr, uint *colInd, T *values, T *X, T *Y, uint N) {
    constexpr auto NumGroups = ThreadsPerBlock / GroupSize;
    auto tid                 = threadIdx.x;
    auto group_id            = tid / GroupSize;
    auto lane_id             = tid % GroupSize;
    auto numel               = divUp(N, gridDim.x);
    uint r_start             = blockIdx.x * numel;
    uint r_end               = min(r_start + numel, N);
    for (uint row = r_start + group_id; row < r_end; row += NumGroups) {
        T sum = 0.0;
        for (uint j = rowPtr[row] + lane_id; j < rowPtr[row + 1]; j += GroupSize) {
            sum += values[j] * X[colInd[j]];
        }
#pragma unroll
        for (int offset = GroupSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        if (lane_id == 0) {
            Y[row] = sum;
        }
    }
}

/// \brief SpMV Baseline implementation on CPU
static void spmv(const uint *rowPtr, const uint *colInd, const double *values, const double *x,
                 double *y, uint numRows) {
    for (uint i = 0; i < numRows; ++i) {
        double sum = 0.0;
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            sum += values[j] * x[colInd[j]];
        }
        y[i] = sum;
    }
}

void run_baseline() {
    CPUTimer timer;
    spmv(mtx.rows, mtx.cols, mtx.vals, x.data(), ref_y.data(), mtx.nrows);
    for (int i = 0; i < warmup_runs; i++) {
        spmv(mtx.rows, mtx.cols, mtx.vals, x.data(), ref_y.data(), mtx.nrows);
    }
    float ms = 0;
    for (int i = 0; i < timed_runs; i++) {
        timer.start();
        spmv(mtx.rows, mtx.cols, mtx.vals, x.data(), ref_y.data(), mtx.nrows);
        timer.stop();
        ms += timer.milliseconds();
    }
    printf("Baseline: %f ms\n", ms / timed_runs);
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file>" << std::endl;
        return 1;
    }
    mtx.loadFromBinary(argv[1]);
    hipDeviceProp_t props;
    check_runtime_api(hipGetDeviceProperties(&props, 0));
    NUM_SM = props.multiProcessorCount;

    x.resize(mtx.ncols);
    ref_y.resize(mtx.nrows);
    Randomizer<double> rand(-1.0, 1.0);
    rand.fill_random(x.data(), x.size());
    memset(ref_y.data(), 0, ref_y.size() * sizeof(double));

    {
        check_runtime_api(hipMalloc(&d_rows, (mtx.nrows + 1) * sizeof(uint)));
        check_runtime_api(hipMalloc(&d_cols, mtx.nnz * sizeof(uint)));
        check_runtime_api(hipMalloc(&d_vals, mtx.nnz * sizeof(double)));
        check_runtime_api(hipMalloc(&d_x, mtx.ncols * sizeof(double)));
        check_runtime_api(hipMalloc(&d_y, mtx.nrows * sizeof(double)));
        check_runtime_api(hipMalloc(&d_rows_i32, (mtx.nrows + 1) * sizeof(int)));
        check_runtime_api(hipMalloc(&d_cols_i32, mtx.nnz * sizeof(int)));
        check_runtime_api(
            hipMemcpy(d_rows, mtx.rows, (mtx.nrows + 1) * sizeof(uint), hipMemcpyHostToDevice));
        check_runtime_api(
            hipMemcpy(d_cols, mtx.cols, mtx.nnz * sizeof(uint), hipMemcpyHostToDevice));
        check_runtime_api(
            hipMemcpy(d_vals, mtx.vals, mtx.nnz * sizeof(double), hipMemcpyHostToDevice));
        check_runtime_api(
            hipMemcpy(d_x, x.data(), mtx.ncols * sizeof(double), hipMemcpyHostToDevice));
        check_runtime_api(hipMemset(d_y, 0, mtx.nrows * sizeof(double)));
    }
    {
        check_runtime_api(hipMalloc(&d_x_f, mtx.ncols * sizeof(float)));
        check_runtime_api(hipMalloc(&d_y_f, mtx.nrows * sizeof(float)));
        check_runtime_api(hipMalloc(&d_vals_f, mtx.nnz * sizeof(float)));
        kCvtPrecision<<<divUp(mtx.ncols, ThreadsPerBlock), ThreadsPerBlock>>>(d_rows_i32, d_rows,
                                                                              mtx.nrows + 1);
        kCvtPrecision<<<divUp(mtx.nnz, ThreadsPerBlock), ThreadsPerBlock>>>(d_cols_i32, d_cols,
                                                                            mtx.nnz);
        kCvtPrecision<<<divUp(mtx.ncols, ThreadsPerBlock), ThreadsPerBlock>>>(d_x_f, d_x,
                                                                              mtx.ncols);
        kCvtPrecision<<<divUp(mtx.nnz, ThreadsPerBlock), ThreadsPerBlock>>>(d_vals_f, d_vals,
                                                                            mtx.nnz);
        check_runtime_api(hipMemset(d_y_f, 0, mtx.nrows * sizeof(float)));
    }

    run_baseline();

    size_t bytes_double = (mtx.nrows + 1) * sizeof(uint) +
                          mtx.nnz * (sizeof(uint) + sizeof(double)) +
                          (mtx.ncols + mtx.nrows) * sizeof(double);
    size_t bytes_float = (mtx.nrows + 1) * sizeof(uint) + mtx.nnz * (sizeof(uint) + sizeof(float)) +
                         (mtx.ncols + mtx.nrows) * sizeof(float);
    size_t flops = 2 * mtx.nnz;
    Profiler profiler(timed_runs, warmup_runs);
    auto check_result_d = [&]() {
        auto ret = validate_all(ref_y.data(), d_y, ref_y.size(), 0, 5e-2);
        check_runtime_api(hipMemset(d_y, 0, mtx.nrows * sizeof(double)));
        return ret;
    };
    auto check_result_f = [&]() {
        auto ret = validate_all(ref_y.data(), d_y_f, ref_y.size(), 0, 5e-2);
        check_runtime_api(hipMemset(d_y_f, 0, mtx.nrows * sizeof(float)));
        return ret;
    };
    
    profiler.add(
        "ValuesDotX-d",
        [&]() {
            constexpr auto VecSize = 2;
            kValuesDotX<ThreadsPerBlock, VecSize>
                <<<divUp(mtx.nnz, ThreadsPerBlock * VecSize), ThreadsPerBlock>>>(d_cols, d_vals,
                                                                                 d_x, d_y,
                                                                                 mtx.nnz);
        },
        nullptr, mtx.nnz * sizeof(double) + mtx.nrows * sizeof(uint), 0);
    profiler.add(
        "ValuesDotX-f",
        [&]() {
            constexpr auto VecSize = 2;
            kValuesDotX<ThreadsPerBlock, VecSize>
                <<<divUp(mtx.nnz, ThreadsPerBlock * VecSize), ThreadsPerBlock>>>(
                    d_cols, d_vals_f, d_x_f, d_y_f, mtx.nnz);
        },
        nullptr, mtx.nnz * sizeof(float) + mtx.nrows * sizeof(uint), 0);

    RocSparseCsrMV<double> rocsparse_d;
    profiler.add(
        "rocSPARSE-dcsrmv",
        [&]() {
            constexpr double alpha = 1.0;
            constexpr double beta  = 0.0;
            rocsparse_d.run(rocsparse_operation_none, mtx.nrows, mtx.ncols, mtx.nnz, &alpha, d_vals,
                            d_rows_i32, d_cols_i32, d_x, &beta, d_y);
        },
        check_result_d, flops, bytes_double);
    RocSparseCsrMV<float> rocsparse_f;
    profiler.add(
        "rocSPARSE-scsrmv",
        [&]() {
            constexpr float alpha = 1.0f;
            constexpr float beta  = 0.0f;
            rocsparse_f.run(rocsparse_operation_none, mtx.nrows, mtx.ncols, mtx.nnz, &alpha,
                            d_vals_f, d_rows_i32, d_cols_i32, d_x_f, &beta, d_y_f);
        },
        check_result_f, flops, bytes_float);
    profiler.add(
        "dSpMV-V4",
        [&]() {
            constexpr auto GroupSize = 4;
            kVectorSubXSpMV<ThreadsPerBlock, GroupSize>
                <<<divUp(mtx.nrows, ThreadsPerBlock / GroupSize), ThreadsPerBlock>>>(
                    d_rows, d_cols, d_vals, d_x, d_y, mtx.nrows);
        },
        check_result_d, flops, bytes_double);

    profiler.add(
        "sSpMV-V4",
        [&]() {
            constexpr auto GroupSize = 4;
            kVectorSubXSpMV<ThreadsPerBlock, GroupSize, float>
                <<<divUp(mtx.nrows, ThreadsPerBlock / GroupSize), ThreadsPerBlock>>>(
                    d_rows, d_cols, d_vals_f, d_x_f, d_y_f, mtx.nrows);
        },
        check_result_f, flops, bytes_float);

    profiler.add(
        "pk-dSpMV-V4",
        [&]() {
            constexpr auto VectorSize = 4;
            pkVectorSubXSpMV<ThreadsPerBlock, VectorSize>
                <<<NUM_SM, ThreadsPerBlock>>>(d_rows, d_cols, d_vals, d_x, d_y, mtx.nrows);
        },
        check_result_d, flops, bytes_double);
    profiler.add(
        "pk-sSpMV-V4",
        [&]() {
            constexpr auto VectorSize = 4;
            pkVectorSubXSpMV<ThreadsPerBlock, VectorSize, float>
                <<<NUM_SM, ThreadsPerBlock>>>(d_rows, d_cols, d_vals_f, d_x_f, d_y_f, mtx.nrows);
        },
        check_result_f, flops, bytes_float);

    profiler.runAll();
    {
        check_runtime_api(hipFree(d_rows));
        check_runtime_api(hipFree(d_cols));
        check_runtime_api(hipFree(d_vals));
        check_runtime_api(hipFree(d_x));
        check_runtime_api(hipFree(d_y));
    }
    {
        check_runtime_api(hipFree(d_x_f));
        check_runtime_api(hipFree(d_vals_f));
        check_runtime_api(hipFree(d_y_f));
    }
}