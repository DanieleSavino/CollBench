#pragma once

#include <stdio.h>
#include <limits.h>

typedef enum {
    CB_SUCCESS = 0,
    CB_ERR_OUT_OF_MEM,
    CB_ERR_OUT_OF_BOUNDS,
    CB_ERR_NULLPTR,
    CB_ERR_INT_OF,
    CB_ERR_MPI,
} CB_Error_t;

#define CB_MALLOC(ptr, size, label) \
do { \
    (ptr) = malloc(size); \
    if(!(ptr)) { \
        err = CB_ERR_OUT_OF_MEM; \
        goto label; \
    } \
} while(0)

#define CB_INT_OF_CHECK(num, label) \
do { \
    if((num) > INT_MAX) { \
        err = CB_ERR_INT_OF; \
        goto label; \
    } \
} while(0)

#define CB_UINT_OF_CHECK(num, label) \
do { \
    if((num) > UINT_MAX) { \
        err = CB_ERR_INT_OF; \
        goto label; \
    } \
} while(0)

#define CB_CHECK(call, label) \
do { \
    err = (call); \
    if(err != CB_SUCCESS) { \
        fprintf(stderr, "[CB ERROR] %s failed: %d\n", #call, err); \
        goto label; \
    } \
} while(0)

#define MPI_CHECK(call, label) \
do { \
    int _mpi_err = (call); \
    if(_mpi_err != MPI_SUCCESS) { \
        err = CB_ERR_MPI; \
        char err_string[MPI_MAX_ERROR_STRING]; \
        int result_len; \
        MPI_Error_string(_mpi_err, err_string, &result_len); \
        fprintf(stderr, "[MPI ERROR] %s failed: %s\n", #call, err_string); \
        goto label; \
    } \
} while(0)
