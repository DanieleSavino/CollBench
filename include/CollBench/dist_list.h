#pragma once

#include "bench.h"
#include "errors.h"
#include <mpi.h>
#include <stddef.h>

#define CB_DEF_INIT_SIZE 10

typedef struct {
    CB_OperationData_t *_buffer;
    size_t len;

    size_t _buff_size;
} CB_DistList_t;

CB_Error_t CB_dlist_init(CB_DistList_t **list, size_t init_size);
CB_Error_t CB_dlist_push(CB_DistList_t * const list, int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t **out);
CB_Error_t CB_dlist_get(const CB_DistList_t *const list, size_t idx, CB_OperationData_t **out);
CB_Error_t CB_dlist_pop(CB_DistList_t * const list, CB_OperationData_t **out);
CB_Error_t CB_dlist_getbyreq(const CB_DistList_t *list, MPI_Request *req, CB_OperationData_t **out);
CB_Error_t CB_dlist_free(CB_DistList_t * const list);

CB_Error_t CB_dlist_pprint(const CB_DistList_t * const list);

CB_Error_t CB_dlist_gather(const CB_DistList_t * const list, MPI_Comm comm, int root, CB_DistList_t **out);

#ifdef CB_PROFILE

#define CB_OP_LWRAP_BLOCKING(list, rank, peer, op_type, algo_idx, call, label)                  \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        CB_CHECK(CB_dlist_push(list, rank, peer, op_type, algo_idx, NULL, &data), label);        \
        CB_op_begin(data);                                                                       \
        MPI_CHECK(call, label);                                                                  \
        CB_op_end(data);                                                                         \
    } while(0)

#define CB_OP_LWRAP_NONBLOCKING(list, rank, peer, op_type, algo_idx, call, req_ref, label)      \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        CB_CHECK(CB_dlist_push(list, rank, peer, op_type, algo_idx, req_ref, &data), label);    \
        CB_op_begin(data);                                                                       \
        MPI_CHECK(call, label);                                                                  \
        CB_op_wait(data);                                                                        \
    } while(0)

#define CB_LSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm)                        \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        CB_OP_LWRAP_NONBLOCKING(_list, rank, dest, CB_OP_SEND, algo_idx,                        \
            MPI_Isend(buff, count, datatype, dest, tag, comm, &_req),                           \
            &_req, _CB_cleanup_label);                                                           \
    } while(0)

#define CB_LRECV(rank, algo_idx, buff, count, datatype, source, tag, comm)                      \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        CB_OP_LWRAP_NONBLOCKING(_list, rank, source, CB_OP_RECV, algo_idx,                      \
            MPI_Irecv(buff, count, datatype, source, tag, comm, &_req),                         \
            &_req, _CB_cleanup_label);                                                           \
    } while(0)

#define CB_ILSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, req_ref)              \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        CB_CHECK(CB_dlist_push(_list, rank, dest, CB_OP_SEND, algo_idx, req_ref, &data),        \
            _CB_cleanup_label);                                                                  \
        CB_op_begin(data);                                                                       \
        MPI_Isend(buff, count, datatype, dest, tag, comm, req_ref);                             \
    } while(0)

#define CB_ILRECV(rank, algo_idx, buff, count, datatype, source, tag, comm, req_ref)            \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        CB_CHECK(CB_dlist_push(_list, rank, source, CB_OP_RECV, algo_idx, req_ref, &data),      \
            _CB_cleanup_label);                                                                  \
        CB_op_begin(data);                                                                       \
        MPI_Irecv(buff, count, datatype, source, tag, comm, req_ref);                           \
    } while(0)

#define CB_LWAIT(req_ref)                                                                        \
    do {                                                                                         \
        CB_OperationData_t *data;                                                                \
        CB_dlist_getbyreq(_list, *(req_ref), &data);                                            \
        CB_op_wait(data);                                                                        \
        MPI_CHECK(MPI_Wait(req_ref, MPI_STATUS_IGNORE), _CB_cleanup_label);                     \
    } while(0)

#define CB_LWAITALL(reqs, buff_len)                                                              \
    do {                                                                                         \
        CB_OperationData_t **buff;                                                               \
        CB_MALLOC(buff, (buff_len) * sizeof(CB_OperationData_t *), _CB_cleanup_label);          \
        for (int i = 0; i < (buff_len); i++) {                                                  \
            CB_dlist_getbyreq(_list, (reqs)[i], &buff[i]);                                      \
        }                                                                                        \
        CB_op_waitall(buff, buff_len);                                                           \
        free(buff);                                                                              \
    } while(0)

#else

#define CB_OP_LWRAP_BLOCKING(list, rank, peer, op_type, algo_idx, call, label)                  \
    do { call; } while(0)

#define CB_OP_LWRAP_NONBLOCKING(list, rank, peer, op_type, algo_idx, call, req_ref, label)      \
    do { call; } while(0)

#define CB_LSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm)                        \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        MPI_Isend(buff, count, datatype, dest, tag, comm, &_req);                               \
        MPI_Wait(&_req, MPI_STATUS_IGNORE);                                                      \
    } while(0)

#define CB_LRECV(rank, algo_idx, buff, count, datatype, source, tag, comm)                      \
    do {                                                                                         \
        MPI_Request _req;                                                                        \
        MPI_Irecv(buff, count, datatype, source, tag, comm, &_req);                             \
        MPI_Wait(&_req, MPI_STATUS_IGNORE);                                                      \
    } while(0)

#define CB_ILSEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, req_ref)              \
    do {                                                                                         \
        MPI_Isend(buff, count, datatype, dest, tag, comm, req_ref);                             \
    } while(0)

#define CB_ILRECV(rank, algo_idx, buff, count, datatype, source, tag, comm, req_ref)            \
    do {                                                                                         \
        MPI_Irecv(buff, count, datatype, source, tag, comm, req_ref);                           \
    } while(0)

#define CB_LWAIT(req_ref)                                                                        \
    do {                                                                                         \
        MPI_Wait(req_ref, MPI_STATUS_IGNORE);                                                    \
    } while(0)

#define CB_LWAITALL(reqs, buff_len)                                                              \
    do {                                                                                         \
        MPI_Waitall(buff_len, reqs, MPI_STATUSES_IGNORE);                                       \
    } while(0)

#endif
