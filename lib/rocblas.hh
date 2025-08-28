#pragma once
// clang-format off
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// clang-format on
#define check_rocblas_api(call)                                                                    \
    do {                                                                                           \
        rocblas_status status = call;                                                              \
        if (status != rocblas_status_success) {                                                    \
            fprintf(stderr, "rocBLAS error in file '%s' in line %i : %d.\n", __FILE__, __LINE__,   \
                    status);                                                                       \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

template <typename T> class RocBlasLv1 {
    using ValueT = T;
    using IndexT = rocblas_int;

  public:
    RocBlasLv1() {
        static_assert(std::is_same_v<ValueT, float> || std::is_same_v<ValueT, double>,
                      "Type not supported");
        check_rocblas_api(rocblas_create_handle(&handle_));
        check_rocblas_api(rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_device));
    };
    ~RocBlasLv1() { check_rocblas_api(rocblas_destroy_handle(handle_)); }

    void nrm2(IndexT n, const ValueT *x, IndexT incx, ValueT *result) {
        if constexpr (std::is_same_v<ValueT, double>)
            check_rocblas_api(rocblas_dnrm2(handle_, n, x, incx, result));
        else if constexpr (std::is_same_v<ValueT, float>) {
            check_rocblas_api(rocblas_snrm2(handle_, n, x, incx, result));
        }
    }
    void scale(IndexT n, const ValueT *alpha, ValueT *x, IndexT incx) {
        if constexpr (std::is_same_v<ValueT, double>)
            check_rocblas_api(rocblas_dscal(handle_, n, alpha, x, incx));
        else if constexpr (std::is_same_v<ValueT, float>) {
            check_rocblas_api(rocblas_sscal(handle_, n, alpha, x, incx));
        }
    }
    void dot(IndexT n, const ValueT *x, IndexT incx, const ValueT *y, IndexT incy, ValueT *result) {
        if constexpr (std::is_same_v<ValueT, double>)
            check_rocblas_api(rocblas_ddot(handle_, n, x, incx, y, incy, result));
        else if constexpr (std::is_same_v<ValueT, float>) {
            check_rocblas_api(rocblas_sdot(handle_, n, x, incx, y, incy, result));
        }
    }
    void axpy(IndexT n, const ValueT *alpha, const ValueT *x, IndexT incx, ValueT *y, IndexT incy) {
        if constexpr (std::is_same_v<ValueT, double>)
            check_rocblas_api(rocblas_daxpy(handle_, n, alpha, x, incx, y, incy));
        else if constexpr (std::is_same_v<ValueT, float>) {
            check_rocblas_api(rocblas_saxpy(handle_, n, alpha, x, incx, y, incy));
        }
    }

  private:
    rocblas_handle handle_;
};
