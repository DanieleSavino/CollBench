/**
 * @file init.h
 * @brief CollBench initialization, finalization, and collective profiling scope macros.
 *
 * Provides CB_init/CB_finalize for MPI datatype lifecycle management, and
 * CB_COLL_START/CB_COLL_END macros that bracket a collective algorithm with
 * list allocation, gather, export, and cleanup. All no-ops when CB_PROFILE
 * is not defined.
 *
 * @author DanieleSavino <savino.2140356@studenti.uniroma1.it>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Daniele - Sapienza Università di Roma
 */
#pragma once
#include "CollBench/errors.h"
#include "CollBench/dist_list.h"
#include "CollBench/export.h"

#ifdef CB_PROFILE

// WARN: CB_init and CB_finalize intentionally do not handle the mpi library;
// separate MPI_Init and MPI_Finalize must be added.

/**
 * @brief Initializes CollBench profiling state.
 *        Must be called after MPI_Init and before any instrumented collective.
 *        Registers the CB_OperationData_t MPI derived datatype.
 * @return CB_SUCCESS or CB_ERR_MPI.
 */
static inline CB_Error_t CB_init(void) {
    CB_Error_t err = CB_SUCCESS;
    CB_CHECK(CB_op_datatype_init(), cleanup);
    cleanup:
        return err;
}

/**
 * @brief Finalizes CollBench profiling state.
 *        Must be called before MPI_Finalize.
 *        Frees the CB_OperationData_t MPI derived datatype.
 * @return CB_SUCCESS or CB_ERR_MPI.
 */
static inline CB_Error_t CB_finalize(void) {
    CB_Error_t err = CB_SUCCESS;
    CB_CHECK(CB_op_datatype_free(), cleanup);
    cleanup:
        return err;
}

// NOTE: The usage of CB_COLL_START without a matching CB_COLL_END or viceversa,
// will be catched at compile time, if CB_PROFILE is set.

/**
 * @brief Opens a profiling scope for a collective algorithm.
 *        Declares err, _list, and _full_list locals and initializes _list.
 *        Must be paired with CB_COLL_END at the end of the same function.
 *        The implicit label _CB_cleanup_label is used by CB_CHECK/MPI_CHECK
 *        inside the collective body to jump to cleanup on error.
 */
#define CB_COLL_START() \
    CB_Error_t err = CB_SUCCESS; \
    CB_DistList_t *_list = NULL, *_full_list = NULL; \
    CB_CHECK(CB_dlist_init(&_list, CB_DEF_INIT_SIZE), _CB_cleanup_label);

/**
 * @brief Closes a profiling scope opened by CB_COLL_START.
 *        Gathers operation records from all ranks onto root, exports to JSON,
 *        then frees both _list and _full_list.
 *        Defines the labels _CB_gather_failed and _CB_cleanup_label which are
 *        used by CB_CHECK/MPI_CHECK in the collective body and in this macro.
 * @param comm     MPI communicator used by the collective.
 * @param rank     Local rank (used to guard root-only export).
 * @param root     Rank that performs the JSON export.
 * @param out_path Path to the output JSON file (root only).
 */
#define CB_COLL_END(comm, rank, root, out_path) \
        CB_CHECK(CB_dlist_gather(_list, comm, root, &_full_list), _CB_gather_failed); \
    _CB_gather_failed: \
        CB_CHECK(CB_dlist_free(_list), _CB_cleanup_label); \
        if(rank == root) \
            CB_CHECK(CB_dlist_export_json(_full_list, out_path), _CB_cleanup_label); \
    _CB_cleanup_label: \
        CB_dlist_free(_full_list)

#else /* CB_PROFILE not defined */

/** @brief No-op in non-profiling builds. */
static inline CB_Error_t CB_init(void) {
    return CB_SUCCESS;
}

/** @brief No-op in non-profiling builds. */
static inline CB_Error_t CB_finalize(void) {
    return CB_SUCCESS;
}

/** @brief No-op in non-profiling builds. */
#define CB_COLL_START() ;

/** @brief No-op in non-profiling builds. */
#define CB_COLL_END(comm, rank, root, out_path) ;

#endif /* CB_PROFILE */
