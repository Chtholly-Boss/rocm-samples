#pragma once
// clang-format off
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

/// \brief Converts a \p rocsparse_status variable to its correspondent string.
inline const char* rocsparse_status_to_string(rocsparse_status status)
{
    switch(status)
    {
        case rocsparse_status_success:                 return "rocsparse_status_success";
        case rocsparse_status_invalid_handle:          return "rocsparse_status_invalid_handle";
        case rocsparse_status_not_implemented:         return "rocsparse_status_not_implemented";
        case rocsparse_status_invalid_pointer:         return "rocsparse_status_invalid_pointer";
        case rocsparse_status_invalid_size:            return "rocsparse_status_invalid_size";
        case rocsparse_status_memory_error:            return "rocsparse_status_memory_error";
        case rocsparse_status_internal_error:          return "rocsparse_status_internal_error";
        case rocsparse_status_invalid_value:           return "rocsparse_status_invalid_value";
        case rocsparse_status_arch_mismatch:           return "rocsparse_status_arch_mismatch";
        case rocsparse_status_zero_pivot:              return "rocsparse_status_zero_pivot";
        case rocsparse_status_not_initialized:         return "rocsparse_status_not_initialized";
        case rocsparse_status_type_mismatch:           return "rocsparse_status_type_mismatch";
        case rocsparse_status_thrown_exception:        return "rocsparse_status_thrown_exception";
// rocSPARSE 3.0 adds new status
#if ROCSPARSE_VERSION_MAJOR >= 3
        case rocsparse_status_continue:                return "rocsparse_status_continue";
#endif
        case rocsparse_status_requires_sorted_storage: return "rocsparse_status_requires_sorted_storage";
        // We do use default because we are not in control of these enumeration values.
        // Ideally this function is something rocsparse would provide
        default:                                       return "<unknown rocsparse_status value>";
    }
}

/// \brief Checks if the provided status code is \p rocsparse_status_success and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define check_rocsparse_api(call)                                                                  \
    do {                                                                                           \
        rocsparse_status status = call;                                                            \
        if (status != rocsparse_status_success) {                                                  \
            fprintf(stderr, "rocSPARSE error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, \
                    rocsparse_status_to_string(status));                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// clang-format on

template <typename T> class RocSparseCsrMV {
    using IndexT = rocsparse_int;
    using ValueT = T;

  public:
    RocSparseCsrMV() {
        check_rocsparse_api(rocsparse_create_handle(&handle_));
        check_rocsparse_api(rocsparse_set_pointer_mode(handle_, rocsparse_pointer_mode_device));
        check_rocsparse_api(rocsparse_create_mat_descr(&descr_));
        check_rocsparse_api(rocsparse_create_mat_info(&info_));
    }
    void run(rocsparse_operation op, IndexT m, IndexT n, IndexT nnz, const ValueT *alpha,
             const ValueT *csr_val, const IndexT *csr_row_ptr, const IndexT *csr_col_ind,
             const ValueT *x, const ValueT *beta, ValueT *y) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float and double are supported");
        if constexpr (std::is_same_v<T, float>) {
            check_rocsparse_api(rocsparse_scsrmv(handle_, op, m, n, nnz, alpha, descr_, csr_val,
                                                 csr_row_ptr, csr_col_ind, info_, x, beta, y));
        } else {
            check_rocsparse_api(rocsparse_dcsrmv(handle_, op, m, n, nnz, alpha, descr_, csr_val,
                                                 csr_row_ptr, csr_col_ind, info_, x, beta, y));
        }
    }
    ~RocSparseCsrMV() {
        rocsparse_destroy_mat_info(info_);
        rocsparse_destroy_mat_descr(descr_);
        rocsparse_destroy_handle(handle_);
    };

  private:
    rocsparse_handle handle_;
    rocsparse_mat_descr descr_;
    rocsparse_mat_info info_;
};
