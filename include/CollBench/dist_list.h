#pragma once

#include "bench.h"
#include "errors.h"
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
CB_Error_t CB_dlist_free(CB_DistList_t * const list);

CB_Error_t CB_dlist_pprint(const CB_DistList_t * const list);

CB_Error_t CB_dlist_gather(const CB_DistList_t * const list, MPI_Comm comm, int root, CB_DistList_t **out);

#define CB_OP_LWRAP_BLOCKING(list, rank, peer, op_type, algo_idx, call, label)        \
do {                                                              \
    CB_OperationData_t *data;                                     \
    CB_CHECK(CB_dlist_push(list, rank, peer, op_type, algo_idx, NULL, &data), label);  \
    CB_op_begin(data);                                            \
    MPI_CHECK(call, label);                                       \
    CB_op_end(data);                                              \
} while(0)

#define CB_OP_LWRAP_NONBLOCKING(list, rank, peer, op_type, algo_idx, call, req_ref, label)        \
do {                                                                          \
    CB_OperationData_t *data;                                                 \
    CB_CHECK(CB_dlist_push(list, rank, peer, op_type, algo_idx, req_ref, &data), label);        \
    CB_op_begin(data);                                                        \
    MPI_CHECK(call, label);                                                   \
    CB_op_wait(data);                                                         \
} while(0)

#define CB_LSEND(list, rank, algo_idx, buff, count, datatype, dest, tag, comm, label) \
do { \
    MPI_Request _req; \
    CB_OP_LWRAP_NONBLOCKING(list, rank, dest, CB_OP_SEND, algo_idx, \
        MPI_Isend(buff, count, datatype, dest, tag, comm, &_req), \
    &_req, label); \
} while(0)

#define CB_LRECV(list, rank, algo_idx, buff, count, datatype, source, tag, comm, label) \
do { \
    MPI_Request _req; \
    CB_OP_LWRAP_NONBLOCKING(list, rank, source, CB_OP_RECV, algo_idx, \
        MPI_Irecv(buff, count, datatype, source, tag, comm, &_req), \
    &_req, label); \
} while(0)
