/**
 * @file bench.h
 * @brief CollBench operation data and profiling macros.
 *
 * Defines CB_OperationData_t, the core per-operation profiling record,
 * along with its associated MPI derived datatype, lifecycle functions,
 * and standalone (non-list) wrapping macros CB_OP_WRAP_BLOCKING,
 * CB_OP_WRAP_NONBLOCKING, CB_SEND, and CB_RECV.
 *
 * For list-based profiling of collective algorithms, see dist_list.h.
 *
 * @author DanieleSavino <savino.2140356@studenti.uniroma1.it>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Daniele - Sapienza Università di Roma
 */

#pragma once
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <time.h>
#include "errors.h"

/** Global MPI derived datatype for CB_OperationData_t. Initialized by CB_op_datatype_init(). */
extern MPI_Datatype _CB_op_datatype;

/** Accessor macro for the global CB_OperationData_t MPI datatype. */
#define CB_OP_DATATYPE (_CB_op_datatype)

/** Enumeration of collective/point-to-point operation types tracked by CollBench. */
typedef enum {
    CB_OP_SEND,
    CB_OP_RECV,
    CB_OP_BCAST,
    CB_OP_REDUCE,
    CB_OP_ALLREDUCE,
    CB_OP_SCATTER,
    CB_OP_GATHER,
    CB_OP_ALLGATHER,
    CB_OP_ALLTOALL,
} CB_OpType_t;

/**
 * @brief Returns a human-readable string for a CB_OpType_t value.
 * @param op_type The operation type.
 * @return A static string such as "send", "recv", "bcast", etc.
 */
const char* CB_optype_str(CB_OpType_t op_type);

/**
 * @brief Per-operation profiling record.
 *
 * Tracks the MPI request handle, communicating ranks, operation type,
 * algorithm index, and three timestamps (post, wait, completion).
 * The req field stores the MPI_Request by value; it is set to
 * MPI_REQUEST_NULL after the operation completes.
 */
typedef struct {
    MPI_Request req;       /**< MPI request handle (value, not pointer). MPI_REQUEST_NULL when complete. */
    int rank;              /**< Rank of the process that posted this operation. */
    int peer;              /**< Peer rank (destination for sends, source for recvs). */
    CB_OpType_t op_type;   /**< Type of operation (send, recv, bcast, ...). */
    size_t algo_idx;       /**< Algorithm step index, used to identify the phase within a collective. */
    uint64_t t_start_ns;   /**< Timestamp when the operation was posted (ns, CLOCK_MONOTONIC). */
    uint64_t t_wait_ns;    /**< Timestamp when the wait was initiated (ns, CLOCK_MONOTONIC). */
    uint64_t t_end_ns;     /**< Timestamp when the operation completed (ns, CLOCK_MONOTONIC). */
} CB_OperationData_t;

/**
 * @brief Allocates and initializes a new CB_OperationData_t.
 * @param rank    Local rank.
 * @param peer    Peer rank.
 * @param op_type Operation type.
 * @param algo_idx Algorithm step index.
 * @param req     Pointer to the MPI_Request to copy (may be NULL for blocking ops).
 * @param data    Output: pointer to the newly allocated record.
 * @return CB_SUCCESS or an error code.
 */
CB_Error_t CB_op_init(int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t **data);

/**
 * @brief Initializes a CB_OperationData_t in-place (no allocation).
 * @param rank    Local rank.
 * @param peer    Peer rank.
 * @param op_type Operation type.
 * @param algo_idx Algorithm step index.
 * @param req     Pointer to the MPI_Request to copy (may be NULL for blocking ops).
 * @param data    Pre-allocated record to initialize.
 * @return CB_SUCCESS or an error code.
 */
CB_Error_t CB_op_init_ext(int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t *data);

/**
 * @brief Frees a heap-allocated CB_OperationData_t.
 * @param data Record to free. Must have been allocated by CB_op_init().
 * @return CB_SUCCESS or CB_ERR_NULLPTR.
 */
CB_Error_t CB_op_free(CB_OperationData_t * const data);

/**
 * @brief Records t_start_ns as the current time.
 * @param data The operation record to stamp.
 * @return CB_SUCCESS or CB_ERR_NULLPTR.
 */
CB_Error_t CB_op_begin(CB_OperationData_t * const data);

/**
 * @brief Waits for the operation to complete and records t_wait_ns and t_end_ns.
 *        Sets req to MPI_REQUEST_NULL on completion.
 * @param data The operation record to wait on.
 * @return CB_SUCCESS or an error code.
 */
CB_Error_t CB_op_wait(CB_OperationData_t * const data);

/**
 * @brief Waits for multiple operations using MPI_Waitall and records timestamps.
 *        Sets req to MPI_REQUEST_NULL on all entries after completion.
 *        Note: t_end_ns is set to the same value for all operations (time of MPI_Waitall return).
 * @param buff     Array of pointers to operation records.
 * @param buff_len Number of entries in buff.
 * @return CB_SUCCESS or an error code.
 */
CB_Error_t CB_op_waitall(CB_OperationData_t ** const buff, size_t buff_len);

/**
 * @brief Records t_end_ns as the current time (for blocking ops without a request).
 * @param data The operation record to stamp.
 * @return CB_SUCCESS or CB_ERR_NULLPTR.
 */
CB_Error_t CB_op_end(CB_OperationData_t * const data);

/**
 * @brief Pretty-prints a CB_OperationData_t record to stdout.
 * @param data The record to print.
 * @return CB_SUCCESS or CB_ERR_NULLPTR.
 */
CB_Error_t CB_op_pprint(const CB_OperationData_t * const data);

/**
 * @brief Creates and commits the MPI derived datatype for CB_OperationData_t.
 *        Must be called after MPI_Init and before any gather operations.
 *        Requires sizeof(MPI_Request) == sizeof(uint64_t) (asserted at compile time).
 * @return CB_SUCCESS or CB_ERR_MPI.
 */
CB_Error_t CB_op_datatype_init(void);

/**
 * @brief Frees the MPI derived datatype created by CB_op_datatype_init().
 * @return CB_SUCCESS or CB_ERR_MPI.
 */
CB_Error_t CB_op_datatype_free(void);

#ifdef CB_PROFILE

/**
 * @brief Wraps a blocking MPI call with profiling.
 *        Allocates a CB_OperationData_t, records start/end timestamps around the call.
 * @param rank     Local rank.
 * @param peer     Peer rank.
 * @param op_type  Operation type.
 * @param algo_idx Algorithm step index.
 * @param call     The blocking MPI call expression.
 * @param ref_data CB_OperationData_t** to receive the allocated record.
 * @param label    Cleanup goto label.
 */
#define CB_OP_WRAP_BLOCKING(rank, peer, op_type, algo_idx, call, ref_data, label)      \
do {                                                              \
    CB_op_init(rank, peer, op_type, algo_idx, NULL, ref_data);                         \
    CB_op_begin(*ref_data);                                       \
    MPI_CHECK(call, label);                                       \
    CB_op_end(*ref_data);                                         \
} while(0)

/**
 * @brief Wraps a nonblocking MPI call with profiling, then immediately waits.
 *        Used for CB_SEND/CB_RECV which are logically blocking but use Isend/Irecv internally.
 *        Note: req_ref must not have side effects (it is evaluated once via a local pointer).
 * @param rank     Local rank.
 * @param peer     Peer rank.
 * @param op_type  Operation type.
 * @param algo_idx Algorithm step index.
 * @param call     The nonblocking MPI call expression (e.g. MPI_Isend(...)).
 * @param ref_data CB_OperationData_t** to receive the allocated record.
 * @param req_ref  Pointer to the MPI_Request written by the nonblocking call.
 * @param label    Cleanup goto label.
 */
#define CB_OP_WRAP_NONBLOCKING(rank, peer, op_type, algo_idx, call, ref_data, req_ref, label) \
do {                                                                     \
    CB_op_init(rank, peer, op_type, algo_idx, req_ref, ref_data);                            \
    CB_op_begin(*ref_data);                                              \
    MPI_CHECK(call, label);                                              \
    CB_op_wait(*ref_data);                                               \
} while(0)

/**
 * @brief Blocking profiled send. Posts MPI_Isend and immediately waits.
 * @param rank     Local rank.
 * @param algo_idx Algorithm step index.
 * @param buff     Send buffer.
 * @param count    Element count.
 * @param datatype MPI datatype.
 * @param dest     Destination rank.
 * @param tag      MPI tag.
 * @param comm     MPI communicator.
 * @param ref_data CB_OperationData_t** to receive the profiling record.
 */
#define CB_SEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, ref_data) \
do { \
    MPI_Request _req; \
    CB_OP_WRAP_NONBLOCKING(rank, dest, CB_OP_SEND, algo_idx, \
        MPI_Isend(buff, count, datatype, dest, tag, comm, &_req), \
    ref_data, &_req, _CB_cleanup_label); \
} while(0)

/**
 * @brief Blocking profiled recv. Posts MPI_Irecv and immediately waits.
 * @param rank     Local rank.
 * @param algo_idx Algorithm step index.
 * @param buff     Receive buffer.
 * @param count    Element count.
 * @param datatype MPI datatype.
 * @param source   Source rank.
 * @param tag      MPI tag.
 * @param comm     MPI communicator.
 * @param ref_data CB_OperationData_t** to receive the profiling record.
 */
#define CB_RECV(rank, algo_idx, buff, count, datatype, source, tag, comm, ref_data) \
do { \
    MPI_Request _req; \
    CB_OP_WRAP_NONBLOCKING(rank, source, CB_OP_RECV, algo_idx, \
        MPI_Irecv(buff, count, datatype, source, tag, comm, &_req), \
    ref_data, &_req, _CB_cleanup_label); \
} while(0)

#else

/** @brief No-op in non-profiling builds. */
#define CB_OP_WRAP_BLOCKING(rank, peer, op_type, algo_idx, call, ref_data, label) \
    do { call; } while(0)

/** @brief No-op in non-profiling builds. */
#define CB_OP_WRAP_NONBLOCKING(rank, peer, op_type, algo_idx, call, ref_data, req_ref, label) \
    do { call; } while(0)

/** @brief Blocking send (no profiling): posts MPI_Isend and immediately waits. */
#define CB_SEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, ref_data) \
    do { \
        MPI_Request _req; \
        MPI_Isend(buff, count, datatype, dest, tag, comm, &_req); \
        MPI_Wait(&_req, MPI_STATUS_IGNORE); \
    } while(0)

/** @brief Blocking recv (no profiling): posts MPI_Irecv and immediately waits. */
#define CB_RECV(rank, algo_idx, buff, count, datatype, source, tag, comm, ref_data) \
    do { \
        MPI_Request _req; \
        MPI_Irecv(buff, count, datatype, source, tag, comm, &_req); \
        MPI_Wait(&_req, MPI_STATUS_IGNORE); \
    } while(0)

#endif
