#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <time.h>
#include "errors.h"

extern MPI_Datatype _CB_op_datatype;
#define CB_OP_DATATYPE (_CB_op_datatype)

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

const char* CB_optype_str(CB_OpType_t op_type);

typedef struct {
    MPI_Request *req;
    int rank;
    int peer;
    CB_OpType_t op_type;
    size_t algo_idx;
    uint64_t t_start_ns;
    uint64_t t_wait_ns;
    uint64_t t_end_ns;
} CB_OperationData_t;

CB_Error_t CB_op_init(int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t **data);
CB_Error_t CB_op_init_ext(int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t *data);
CB_Error_t CB_op_free(CB_OperationData_t * const data);
CB_Error_t CB_op_begin(CB_OperationData_t * const data);
CB_Error_t CB_op_wait(CB_OperationData_t * const data);
CB_Error_t CB_op_end(CB_OperationData_t * const data);
CB_Error_t CB_op_pprint(const CB_OperationData_t * const data);

CB_Error_t CB_op_datatype_init(void);
CB_Error_t CB_op_datatype_free(void);

#define CB_OP_WRAP_BLOCKING(rank, peer, op_type, algo_idx, call, ref_data, label)      \
do {                                                              \
    CB_op_init(rank, peer, op_type, algo_idx, NULL, ref_data);                         \
    CB_op_begin(*ref_data);                                       \
    MPI_CHECK(call, label);                                       \
    CB_op_end(*ref_data);                                         \
} while(0)

#define CB_OP_WRAP_NONBLOCKING(rank, peer, op_type, algo_idx, call, ref_data, req_ref, label) \
do {                                                                     \
    CB_op_init(rank, peer, op_type, algo_idx, &req_ref, ref_data);                            \
    CB_op_begin(*ref_data);                                              \
    MPI_CHECK(call, label);                                              \
    CB_op_wait(*ref_data);                                               \
} while(0)

#define CB_SEND(rank, algo_idx, buff, count, datatype, dest, tag, comm, ref_data, label) \
do { \
    MPI_Request _req; \
    CB_OP_WRAP_NONBLOCKING(rank, dest, CB_OP_SEND, algo_idx \
        MPI_Isend(buff, count, datatype, dest, tag, comm, &_req), \
    ref_data, &_req, label); \
} while(0)

#define CB_RECV(rank, algo_idx, buff, count, datatype, source, tag, comm, ref_data, label) \
do { \
    MPI_Request _req; \
    CB_OP_WRAP_NONBLOCKING(rank, source, CB_OP_SEND, algo_idx \
        MPI_Irecv(buff, count, datatype, source, tag, comm, &_req), \
    ref_data, &_req, label); \
} while(0)

