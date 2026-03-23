#pragma once

#include <stdio.h>

typedef enum {
    CB_SUCCESS = 0,
    CB_ERR_OUT_OF_MEM,
    CB_ERR_OUT_OF_BOUNDS,
    CB_ERR_NULLPTR,
    CB_ERR_MPI,
} CB_Error_t;

#define CB_CHECK(call)                                          \
do {                                                            \
    CB_Error_t err = (call);                                    \
    if (err != CB_SUCCESS) {                                    \
        fprintf(stderr, "[CB ERROR] %s failed: %d\n", #call, err); \
        exit(err);                                              \
    }                                                           \
} while(0)

#define MPI_CHECK(call)                                         \
do {                                                            \
    int _mpi_err = (call);                                      \
    if (_mpi_err != MPI_SUCCESS) {                              \
        char err_string[MPI_MAX_ERROR_STRING];                  \
        int result_len;                                         \
        MPI_Error_string(_mpi_err, err_string, &result_len);    \
        fprintf(stderr, "[MPI ERROR] %s failed: %s\n", #call, err_string); \
        return CB_ERR_MPI;                                      \
    }                                                           \
} while(0)
