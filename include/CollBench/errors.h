#pragma once

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

typedef enum {
    CB_SUCCESS = 0,
    CB_ERR_OUT_OF_MEM,
    CB_ERR_OUT_OF_BOUNDS,
    CB_ERR_NULLPTR,
    CB_ERR_INT_OF,
    CB_ERR_IO,
    CB_ERR_INVALID_ARG,
    CB_ERR_MPI,
} CB_Error_t;

static inline char* CB_strerr(CB_Error_t err) {
    switch (err) {
        case CB_SUCCESS:           return "Success";
        case CB_ERR_OUT_OF_MEM:    return "Out of memory";
        case CB_ERR_OUT_OF_BOUNDS: return "Index out of bounds";
        case CB_ERR_NULLPTR:       return "Null pointer";
        case CB_ERR_INT_OF:        return "Integer overflow";
        case CB_ERR_IO:            return "I/O error";
        case CB_ERR_INVALID_ARG:   return "Invalid argument";
        case CB_ERR_MPI:           return "MPI error";
        default:                   return "Unknown error";
    }
}

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
        fprintf(stderr, "[CB ERROR] %s in file: %s at line %d failed: %s\n", #call, __FILE__, __LINE__, CB_strerr(err)); \
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
