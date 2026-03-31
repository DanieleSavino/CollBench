/**
 * @file errors.h
 * @brief Error codes, error utilities, and safety macros for CollBench.
 *
 * Defines CB_Error_t and helper macros for memory allocation (CB_MALLOC),
 * overflow checks (CB_INT_OF_CHECK, CB_UINT_OF_CHECK), error propagation
 * (CB_CHECK), and MPI error handling (MPI_CHECK). All macros assume a local
 * CB_Error_t err variable and a cleanup goto label are in scope.
 *
 * @author DanieleSavino <savino.2140356@studenti.uniroma1.it>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Daniele - Sapienza Università di Roma
 */
#pragma once
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Return codes used throughout CollBench.
 *        CB_SUCCESS is guaranteed to be 0 so it can be used as a boolean.
 */
typedef enum {
    CB_SUCCESS = 0,        /**< Operation completed successfully. */
    CB_ERR_OUT_OF_MEM,     /**< malloc/realloc returned NULL. */
    CB_ERR_OUT_OF_BOUNDS,  /**< Index exceeded container size. */
    CB_ERR_NULLPTR,        /**< A required pointer argument was NULL. */
    CB_ERR_INT_OF,         /**< Integer overflow detected. */
    CB_ERR_IO,             /**< I/O operation failed. */
    CB_ERR_INVALID_ARG,    /**< Argument value is invalid (e.g. not found). */
    CB_ERR_MPI,            /**< An MPI call returned a non-MPI_SUCCESS code. */
} CB_Error_t;

/**
 * @brief Returns a human-readable string for a CB_Error_t value.
 * @param err The error code.
 * @return A static string describing the error.
 */
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

/**
 * @brief Allocates memory and jumps to label on failure.
 *        Sets err = CB_ERR_OUT_OF_MEM before jumping.
 * @param ptr   The pointer to assign the allocation to.
 * @param size  Number of bytes to allocate.
 * @param label Cleanup goto label.
 */
#define CB_MALLOC(ptr, size, label) \
do { \
    (ptr) = malloc(size); \
    if(!(ptr)) { \
        err = CB_ERR_OUT_OF_MEM; \
        goto label; \
    } \
} while(0)

/**
 * @brief Checks that num fits in a signed int, jumps to label on overflow.
 *        Sets err = CB_ERR_INT_OF before jumping.
 * @param num   The value to check (must be comparable to INT_MAX).
 * @param label Cleanup goto label.
 */
#define CB_INT_OF_CHECK(num, label) \
do { \
    if((num) > INT_MAX) { \
        err = CB_ERR_INT_OF; \
        goto label; \
    } \
} while(0)

/**
 * @brief Checks that num fits in an unsigned int, jumps to label on overflow.
 *        Sets err = CB_ERR_INT_OF before jumping.
 * @param num   The value to check (must be comparable to UINT_MAX).
 * @param label Cleanup goto label.
 */
#define CB_UINT_OF_CHECK(num, label) \
do { \
    if((num) > UINT_MAX) { \
        err = CB_ERR_INT_OF; \
        goto label; \
    } \
} while(0)

/**
 * @brief Calls a CB_Error_t-returning function, prints an error and jumps on failure.
 *        Assigns the return value to the local err variable.
 * @param call  The function call expression returning CB_Error_t.
 * @param label Cleanup goto label.
 */
#define CB_CHECK(call, label) \
do { \
    err = (call); \
    if(err != CB_SUCCESS) { \
        fprintf(stderr, "[CB ERROR] %s in file: %s at line %d failed: %s\n", #call, __FILE__, __LINE__, CB_strerr(err)); \
        goto label; \
    } \
} while(0)

/**
 * @brief Calls an MPI function, prints a detailed error and jumps on failure.
 *        Sets err = CB_ERR_MPI and prints the MPI error string via MPI_Error_string.
 * @param call  The MPI call expression returning an MPI error code.
 * @param label Cleanup goto label.
 */
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
