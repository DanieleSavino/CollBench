/**
 * @file dist_list.h
 * @brief Distributed operation list and profiling macros for CollBench.
 *
 * Provides CB_DistList_t, a dynamic array of CB_OperationData_t records,
 * along with macros for wrapping MPI point-to-point calls with optional
 * profiling (enabled by -DCB_PROFILE).
 *
 * @author DanieleSavino <savino.2140356@studenti.uniroma1.it>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Daniele - Sapienza Universita' di Roma
 */
#pragma once

#include "bench.h"
#include "errors.h"
#include <mpi.h>
#include <stddef.h>

/** Default initial capacity of a CB_DistList_t buffer. */
#define CB_DEF_INIT_SIZE 10

/**
 * @brief Dynamic array of CB_OperationData_t records, one per MPI operation.
 *
 * Used to collect per-operation profiling data during a collective algorithm.
 * The buffer doubles in capacity when full (see CB_dlist_push).
 * After the collective completes, CB_dlist_gather aggregates records from
 * all ranks onto the root for export.
 */
typedef struct {
    CB_OperationData_t *_buffer; /**< Heap-allocated array of operation records. */
    size_t len;                  /**< Number of valid records currently in the buffer. */
    size_t _buff_size;           /**< Current allocated capacity of _buffer. */
} CB_DistList_t;

/**
 * @brief Allocates and initializes an empty CB_DistList_t.
 * @param list      Output: pointer to the newly allocated list.
 * @param init_size Initial buffer capacity (number of records).
 * @return CB_SUCCESS or CB_ERR_OUT_OF_MEM.
 */
CB_Error_t CB_dlist_init(CB_DistList_t **list, size_t init_size);

/**
 * @brief Appends a new operation record to the list.
 *        Doubles the buffer capacity if needed.
 *        The MPI_Request is copied by value from *req.
 * @param list     The list to append to.
 * @param rank     Local rank.
 * @param peer     Peer rank.
 * @param op_type  Operation type.
 * @param algo_idx Algorithm step index.
 * @param req      Pointer to the MPI_Request to copy (may be NULL for blocking ops).
 * @param out      Output: pointer to the newly appended record.
 * @return CB_SUCCESS or an error code.
 */
CB_Error_t CB_dlist_push(CB_DistList_t * const list, int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t **out);

/**
 * @brief Returns a pointer to the record at the given index.
 * @param list The list to index into.
 * @param idx  Zero-based index.
 * @param out  Output: pointer to the record at idx.
 * @return CB_SUCCESS or CB_ERR_OUT_OF_BOUNDS.
 */
CB_Error_t CB_dlist_get(const CB_DistList_t *const list, size_t idx, CB_OperationData_t **out);

/**
 * @brief Removes and optionally returns the last record in the list.
 *        NOTE: The result can be discarded setting out to NULL
 * @param list The list to pop from.
 * @param out  Output: heap-allocated copy of the popped record, or NULL to discard.
 *             Caller is responsible for freeing *out if non-NULL.
 * @return CB_SUCCESS or CB_ERR_OUT_OF_BOUNDS.
 */
CB_Error_t CB_dlist_pop(CB_DistList_t * const list, CB_OperationData_t **out);

/**
 * @brief Finds the first record whose req field matches *req.
 *        Comparison is by MPI_Request value, not pointer.
 *        WARN: The MPI_Request (aka pointer to some mpi struct) may not be a key
 *        as mpi might reuse memory regions after freeing old requests
 * @param list The list to search.
 * @param req  Pointer to the MPI_Request value to search for.
 * @param out  Output: pointer to the matching record, or NULL if not found.
 * @return CB_SUCCESS, CB_ERR_INVALID_ARG (not found), or CB_ERR_NULLPTR.
 */
CB_Error_t CB_dlist_getbyreq(const CB_DistList_t *list, MPI_Request *req, CB_OperationData_t **out);

/**
 * @brief Frees a CB_DistList_t and its internal buffer.
 * @param list The list to free.
 * @return CB_SUCCESS or CB_ERR_NULLPTR.
 */
CB_Error_t CB_dlist_free(CB_DistList_t * const list);

/**
 * @brief Pretty-prints all records in the list to stdout.
 * @param list The list to print.
 * @return CB_SUCCESS or CB_ERR_NULLPTR.
 */
CB_Error_t CB_dlist_pprint(const CB_DistList_t * const list);

/**
 * @brief Gathers operation records from all ranks onto root using MPI_Gatherv.
 *        Non-root ranks return with *out == NULL.
 *        WARN: req fields in the gathered list are local handles from remote
 *        processes and must not be dereferenced or used for lookup on the root.
 * @param list  The local list to gather from.
 * @param comm  MPI communicator.
 * @param root  Rank that receives all records.
 * @param out   Output: pointer to the gathered list (root only), or NULL.
 * @return CB_SUCCESS or an error code.
 */
CB_Error_t CB_dlist_gather(const CB_DistList_t * const list, MPI_Comm comm, int root, CB_DistList_t **out);

#ifdef CB_PROFILE

/**
 * @brief Wraps a blocking MPI call with list-based profiling.
 *        Pushes a record into list, records start/end timestamps around call.
 * @param list     The CB_DistList_t to record into.
 * @param rank     Local rank.
 * @param peer     Peer rank.
 * @param op_type  Operation type.
 * @param algo_idx Algorithm step index.
 * @param call     The blocking MPI call expression.
 * @param label    Cleanup goto label.
 */
#define CB_OP_LWRAP_BLOCKING(list, rank, peer, op_type, algo_idx, call, label)                  \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        CB_CHECK(CB_dlist_push(list, rank, peer, op_type, algo_idx, NULL, &data), label);        \
        CB_op_begin(data);                                                                       \
        MPI_CHECK(call, label);                                                                  \
        CB_op_end(data);                                                                         \
    } while(0)

/**
 * @brief Wraps a nonblocking MPI call with list-based profiling, then immediately waits.
 *        The MPI call is issued first so the request handle is valid before being stored.
 *        Used by CB_LSEND/CB_LRECV which are logically blocking.
 * @param list     The CB_DistList_t to record into.
 * @param rank     Local rank.
 * @param peer     Peer rank.
 * @param op_type  Operation type.
 * @param algo_idx Algorithm step index.
 * @param call     The nonblocking MPI call expression.
 * @param req_ref  Pointer to the MPI_Request written by call. Must not have side effects.
 * @param label    Cleanup goto label.
 */
#define CB_OP_LWRAP_NONBLOCKING(list, rank, peer, op_type, algo_idx, call, req_ref, label)      \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        MPI_CHECK(call, label);                                                                  \
        CB_CHECK(CB_dlist_push(list, rank, peer, op_type, algo_idx, req_ref, &data), label);    \
        CB_op_begin(data);                                                                       \
        CB_op_wait(data);                                                                        \
    } while(0)

/**
 * @brief Blocking profiled send. Posts MPI_Isend and immediately waits.
 *        Records the operation into the implicit _list.
 * @param rank     Local rank.
 * @param algo_idx Algorithm step index.
 * @param buff     Send buffer.
 * @param count    Element count.
 * @param datatype MPI datatype.
 * @param dest     Destination rank.
 * @param tag      MPI tag.
 * @param comm     MPI communicator.
 */
#define CB_LSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm)                        \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        CB_OP_LWRAP_NONBLOCKING(_list, rank, dest, CB_OP_SEND, algo_idx,                        \
            MPI_Isend(buff, count, datatype, dest, tag, comm, &_req),                           \
            &_req, _CB_cleanup_label);                                                           \
    } while(0)

/**
 * @brief Blocking profiled recv. Posts MPI_Irecv and immediately waits.
 *        Records the operation into the implicit _list.
 * @param rank     Local rank.
 * @param algo_idx Algorithm step index.
 * @param buff     Receive buffer.
 * @param count    Element count.
 * @param datatype MPI datatype.
 * @param source   Source rank.
 * @param tag      MPI tag.
 * @param comm     MPI communicator.
 */
#define CB_LRECV(rank, algo_idx, buff, count, datatype, source, tag, comm)                      \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        CB_OP_LWRAP_NONBLOCKING(_list, rank, source, CB_OP_RECV, algo_idx,                      \
            MPI_Irecv(buff, count, datatype, source, tag, comm, &_req),                         \
            &_req, _CB_cleanup_label);                                                           \
    } while(0)

/**
 * @brief Nonblocking profiled send. Posts MPI_Isend and returns immediately.
 *        WARN: The request is stored in the operation data and must be waited on later via CB_LWAIT.
 *        A manual MPI_Wait will break the CollBench data structure.
 *        NOTE: req_ref is captured into a local pointer to prevent double-evaluation of
 *        expressions with side effects (e.g. &reqs[req_idx++]).
 * @param rank     Local rank.
 * @param algo_idx Algorithm step index.
 * @param buff     Send buffer.
 * @param count    Element count.
 * @param datatype MPI datatype.
 * @param dest     Destination rank.
 * @param tag      MPI tag.
 * @param comm     MPI communicator.
 * @param req_ref  Pointer to MPI_Request to receive the handle.
 */
#define CB_ILSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, req_ref)              \
    do {                                                                                         \
        MPI_Request *req = req_ref;                                                              \
        CB_OperationData_t *data;                                                                \
        MPI_CHECK(MPI_Isend(buff, count, datatype, dest, tag, comm, req),                       \
            _CB_cleanup_label);                                                                  \
        CB_CHECK(CB_dlist_push(_list, rank, dest, CB_OP_SEND, algo_idx, req, &data),            \
            _CB_cleanup_label);                                                                  \
        CB_op_begin(data);                                                                       \
    } while(0)

/**
 * @brief Nonblocking profiled recv. Posts MPI_Irecv and returns immediately.
 *        WARN: The request is stored in the operation data and must be waited on later via CB_LWAIT.
 *        A manual MPI_Wait will break the CollBench data structure.
 *        NOTE: req_ref is captured into a local pointer to prevent double-evaluation of
 *        expressions with side effects (e.g. &reqs[req_idx++]).
 * @param rank     Local rank.
 * @param algo_idx Algorithm step index.
 * @param buff     Receive buffer.
 * @param count    Element count.
 * @param datatype MPI datatype.
 * @param source   Source rank.
 * @param tag      MPI tag.
 * @param comm     MPI communicator.
 * @param req_ref  Pointer to MPI_Request to receive the handle.
 */
#define CB_ILRECV(rank, algo_idx, buff, count, datatype, source, tag, comm, req_ref)            \
    do {                                                                                         \
        MPI_Request *req = req_ref;                                                              \
        CB_OperationData_t *data;                                                                \
        MPI_CHECK(MPI_Irecv(buff, count, datatype, source, tag, comm, req),                     \
            _CB_cleanup_label);                                                                  \
        CB_CHECK(CB_dlist_push(_list, rank, source, CB_OP_RECV, algo_idx, req, &data),          \
            _CB_cleanup_label);                                                                  \
        CB_op_begin(data);                                                                       \
    } while(0)

/**
 * @brief Waits on a single previously posted nonblocking operation identified by req_ref.
 *        Looks up the matching record in _list and calls CB_op_wait.
 * @param req_ref Pointer to the MPI_Request to wait on.
 */
#define CB_LWAIT(req_ref)                                                                        \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
            \
        /** \
        * FIXME: If a request gets allocated, then freed without setting req to null \
        * then mpi reuses that memory address there might be a conflict \
        * this should mitigated by CB_op_wait setting req to NULL, but it's not 100% safe \
        * this design choice is aimed at closely match the mpi api \
        * CB_CHECK(CB_dlist_getbyreq(_list, req_ref, &data), _CB_cleanup_label);                  \
        */ \
            \
        CB_op_wait(data);                                                                        \
    } while(0)

/**
 * @brief Waits on all previously posted nonblocking operations in reqs[0..buff_len-1].
 *        Looks up each request in _list by value and calls CB_op_waitall to record timestamps.
 *        NOTE: All requests must have unique, non-null MPI_Request values at lookup time.
 *        Requests completed eagerly by MPI (set to NULL) may cause lookup failure.
 * @param reqs     Array of MPI_Request handles.
 * @param buff_len Number of requests in reqs.
 */
#define CB_LWAITALL(reqs, buff_len)                                                              \
    do {                                                                                         \
        CB_OperationData_t **buff;                                                               \
        CB_MALLOC(buff, (buff_len) * sizeof(CB_OperationData_t *), _CB_cleanup_label);          \
        for (int i = 0; i < (buff_len); i++) {                                                  \
            CB_OperationData_t *data;                                                            \
            /**
            * FIXME: If a request gets allocated, then freed without setting req to null \
            * then mpi reuses that memory address there might be a conflict \
            * this should mitigated by CB_op_waitall setting all reqs to NULL, but it's not 100% safe \
            * this design choice is aimed at closely match the mpi api \
            */ \
            CB_CHECK(CB_dlist_getbyreq(_list, &(reqs)[i], &data), _CB_cleanup_label);           \
                \
            buff[i] = data;                                                                      \
        }                                                                                        \
        CB_op_waitall(buff, buff_len);                                                           \
        free(buff);                                                                              \
    } while(0)

#else /* CB_PROFILE not defined - all macros reduce to bare MPI calls */

/** @brief No-op in non-profiling builds. */
#define CB_OP_LWRAP_BLOCKING(list, rank, peer, op_type, algo_idx, call, label)                  \
    do { call; } while(0)

/** @brief No-op in non-profiling builds. */
#define CB_OP_LWRAP_NONBLOCKING(list, rank, peer, op_type, algo_idx, call, req_ref, label)      \
    do { call; } while(0)

/** @brief Blocking send (no profiling): posts MPI_Isend and immediately waits. */
#define CB_LSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm)                        \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        MPI_Isend(buff, count, datatype, dest, tag, comm, &_req);                               \
        MPI_Wait(&_req, MPI_STATUS_IGNORE);                                                      \
    } while(0)

/** @brief Blocking recv (no profiling): posts MPI_Irecv and immediately waits. */
#define CB_LRECV(rank, algo_idx, buff, count, datatype, source, tag, comm)                      \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        MPI_Irecv(buff, count, datatype, source, tag, comm, &_req);                             \
        MPI_Wait(&_req, MPI_STATUS_IGNORE);                                                      \
    } while(0)

/** @brief Nonblocking send (no profiling): posts MPI_Isend, caller waits later. */
#define CB_ILSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, req_ref)              \
    do {                                                                                         \
        MPI_Isend(buff, count, datatype, dest, tag, comm, req_ref);                             \
    } while(0)

/** @brief Nonblocking recv (no profiling): posts MPI_Irecv, caller waits later. */
#define CB_ILRECV(rank, algo_idx, buff, count, datatype, source, tag, comm, req_ref)            \
    do {                                                                                         \
        MPI_Irecv(buff, count, datatype, source, tag, comm, req_ref);                           \
    } while(0)

/** @brief Waits on a single request (no profiling). */
#define CB_LWAIT(req_ref)                                                                        \
    do {                                                                                         \
        MPI_Wait(req_ref, MPI_STATUS_IGNORE);                                                    \
    } while(0)

/** @brief Waits on all requests in reqs[0..buff_len-1] (no profiling). */
#define CB_LWAITALL(reqs, buff_len)                                                              \
    do {                                                                                         \
        MPI_Waitall(buff_len, reqs, MPI_STATUSES_IGNORE);                                       \
    } while(0)

#endif /* CB_PROFILE */
