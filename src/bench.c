#include "CollBench/bench.h"
#include "CollBench/errors.h"
#include <stdio.h>
#include <inttypes.h>
#include <mpi.h>

static inline uint64_t getCurrentTimeNS(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

CB_Error_t CB_op_init(MPI_Request *req, size_t algo_idx, CB_OperationData_t **data) {
    *data = (CB_OperationData_t*)malloc(sizeof(CB_OperationData_t));
    if(! (*data)) {
        return CB_ERR_OUT_OF_MEM;
    }

    CB_op_init_ext(req, algo_idx, *data);

    return CB_SUCCESS;
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
    if(!data) {
        return CB_ERR_NULLPTR;
    }

    data->t_wait_ns = getCurrentTimeNS();

    MPI_CHECK(MPI_Wait(data->req, MPI_STATUS_IGNORE));

    data->t_end_ns = getCurrentTimeNS();

    return CB_SUCCESS;
}

CB_Error_t CB_op_end(CB_OperationData_t * const data) {
    if(!data) {
        return CB_ERR_NULLPTR;
    }

    data->t_end_ns = getCurrentTimeNS();

    return CB_SUCCESS;
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
