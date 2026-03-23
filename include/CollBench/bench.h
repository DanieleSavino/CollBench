#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <time.h>
#include "errors.h"

typedef struct {
    MPI_Request *req;
    size_t algo_idx;
    uint64_t t_start_ns;
    uint64_t t_wait_ns;
    uint64_t t_end_ns;
} CB_OperationData_t;

CB_Error_t CB_op_init(MPI_Request *req, size_t algo_idx, CB_OperationData_t **data);
CB_Error_t CB_op_init_ext(MPI_Request *req, size_t algo_idx, CB_OperationData_t *data);
CB_Error_t CB_op_free(CB_OperationData_t * const data);
CB_Error_t CB_op_begin(CB_OperationData_t * const data);
CB_Error_t CB_op_wait(CB_OperationData_t * const data);
CB_Error_t CB_op_end(CB_OperationData_t * const data);
CB_Error_t CB_op_pprint(const CB_OperationData_t * const data);

#define CB_OP_WRAP_BLOCKING(algo_idx, call, ref_data)      \
do {                                                              \
    CB_op_init(NULL, algo_idx, ref_data);                   \
    CB_op_begin(*ref_data);                                       \
    MPI_CHECK(call);                                                         \
    CB_op_end(*ref_data);                                         \
} while(0)

#define CB_OP_WRAP_NONBLOCKING(algo_idx, call, ref_data, req_ref) \
do {                                                                     \
    CB_op_init(&req_ref, algo_idx, ref_data);                       \
    CB_op_begin(*ref_data);                                               \
    MPI_CHECK(call);                                                         \
    CB_op_wait(*ref_data);                                                \
} while(0)
