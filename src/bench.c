#include "CollBench/bench.h"
#include "CollBench/errors.h"
#include <stdio.h>
#include <inttypes.h>
#include <mpi.h>

MPI_Datatype _CB_op_datatype = MPI_DATATYPE_NULL;

static inline uint64_t getCurrentTimeNS(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

CB_Error_t CB_op_init(MPI_Request *req, size_t algo_idx, CB_OperationData_t **data) {
    CB_Error_t err = CB_SUCCESS;
    CB_MALLOC(*data, sizeof(CB_OperationData_t), cleanup);

    CB_CHECK(CB_op_init_ext(req, algo_idx, *data), cleanup);

    cleanup:
        return err;
}

CB_Error_t CB_op_init_ext(MPI_Request *req, size_t algo_idx, CB_OperationData_t *data) {
    data->req = req;
    data->algo_idx = algo_idx;
    data->t_start_ns = 0;
    data->t_wait_ns = 0;
    data->t_end_ns = 0;

    return CB_SUCCESS;
}

CB_Error_t CB_op_free(CB_OperationData_t * const data) {
    if(!data) {
        return CB_ERR_NULLPTR;
    }

    free(data);

    return CB_SUCCESS;
}

CB_Error_t CB_op_begin(CB_OperationData_t * const data) {
    if(!data) {
        return CB_ERR_NULLPTR;
    }

    data->t_start_ns = getCurrentTimeNS();

    return CB_SUCCESS;
}

CB_Error_t CB_op_wait(CB_OperationData_t * const data) {
    CB_Error_t err = CB_SUCCESS;

    if(!data) {
        return CB_ERR_NULLPTR;
    }

    data->t_wait_ns = getCurrentTimeNS();

    MPI_CHECK(MPI_Wait(data->req, MPI_STATUS_IGNORE), cleanup);

    data->t_end_ns = getCurrentTimeNS();

    cleanup:
        return err;
}

CB_Error_t CB_op_end(CB_OperationData_t * const data) {
    if(!data) {
        return CB_ERR_NULLPTR;
    }

    data->t_end_ns = getCurrentTimeNS();

    return CB_SUCCESS;
}

CB_Error_t CB_op_datatype_init(void) {
    CB_Error_t err = CB_SUCCESS;

    static const struct {
        MPI_Aint     offset;
        MPI_Datatype type;
    } fields[] = {
        { offsetof(CB_OperationData_t, req),        MPI_UINT64_T }, /* MPI_Request* as opaque 8-byte value */
        { offsetof(CB_OperationData_t, algo_idx),   MPI_UINT64_T },
        { offsetof(CB_OperationData_t, t_start_ns), MPI_UINT64_T },
        { offsetof(CB_OperationData_t, t_wait_ns),  MPI_UINT64_T },
        { offsetof(CB_OperationData_t, t_end_ns),   MPI_UINT64_T },
    };

    enum { NFIELDS = sizeof(fields) / sizeof(fields[0]) };

    int          lengths[NFIELDS];
    MPI_Aint     displacements[NFIELDS];
    MPI_Datatype types[NFIELDS];

    for (int i = 0; i < NFIELDS; i++) {
        lengths[i]       = 1;
        displacements[i] = fields[i].offset;
        types[i]         = fields[i].type;
    }

    MPI_Datatype tmp;
    MPI_CHECK(MPI_Type_create_struct(NFIELDS, lengths, displacements, types, &tmp), cleanup);
    MPI_CHECK(MPI_Type_create_resized(tmp, 0, sizeof(CB_OperationData_t), &_CB_op_datatype), cleanup);
    MPI_CHECK(MPI_Type_free(&tmp), cleanup);
    MPI_CHECK(MPI_Type_commit(&_CB_op_datatype), cleanup);

    cleanup:
        return err;
}

CB_Error_t CB_op_datatype_free(void) {
    CB_Error_t err = CB_SUCCESS;
    MPI_CHECK(MPI_Type_free(&_CB_op_datatype), cleanup);

    cleanup:
        return err;
}

CB_Error_t CB_op_pprint(const CB_OperationData_t * const data) {
    if (!data) {
        return CB_ERR_NULLPTR;
    }

    printf("=== CB Operation Data ===\n");
    printf("Algo index : %zu\n", data->algo_idx);

    if (data->req) {
        printf("MPI Req    : %p\n", (void*)data->req);
    } else {
        printf("MPI Req    : (blocking/no request)\n");
    }

    // timestamps
    printf("t_start_ns : %" PRIu64 " ns\n", data->t_start_ns);
    printf("t_wait_ns  : %" PRIu64 " ns\n", data->t_wait_ns);
    printf("t_end_ns   : %" PRIu64 " ns\n", data->t_end_ns);

    // derived durations
    if (data->t_start_ns && data->t_end_ns) {
        uint64_t total_ns = data->t_end_ns - data->t_start_ns;
        printf("Total time : %" PRIu64 " ns (%.3f ms)\n", total_ns, total_ns / 1e6);
    }

    if (data->t_start_ns && data->t_wait_ns) {
        uint64_t pre_wait_ns = data->t_wait_ns - data->t_start_ns;
        printf("Pre-wait   : %" PRIu64 " ns (%.3f ms)\n", pre_wait_ns, pre_wait_ns / 1e6);
    }

    if (data->t_wait_ns && data->t_end_ns) {
        uint64_t wait_to_end_ns = data->t_end_ns - data->t_wait_ns;
        printf("Post-wait  : %" PRIu64 " ns (%.3f ms)\n", wait_to_end_ns, wait_to_end_ns / 1e6);
    }

    printf("=========================\n");

    return CB_SUCCESS;
}
