#include "CollBench/bench.h"
#include "CollBench/errors.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <inttypes.h>
#include <mpi.h>

MPI_Datatype _CB_op_datatype = MPI_DATATYPE_NULL;

static inline uint64_t getCurrentTimeNS(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

const char* CB_optype_str(CB_OpType_t op) {
    switch (op) {
        case CB_OP_SEND:      return "send";
        case CB_OP_RECV:      return "recv";
        case CB_OP_BCAST:     return "bcast";
        case CB_OP_REDUCE:    return "reduce";
        case CB_OP_ALLREDUCE: return "allreduce";
        case CB_OP_SCATTER:   return "scatter";
        case CB_OP_GATHER:    return "gather";
        case CB_OP_ALLGATHER: return "allgather";
        case CB_OP_ALLTOALL:  return "alltoall";
        default:              return "unknown";
    }
}

CB_Error_t CB_op_init(int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t **data) {
    CB_Error_t err = CB_SUCCESS;
    CB_MALLOC(*data, sizeof(CB_OperationData_t), cleanup);

    CB_CHECK(CB_op_init_ext(rank, peer, op_type, algo_idx, req, *data), cleanup);

    cleanup:
        return err;
}

CB_Error_t CB_op_init_ext(int rank, int peer, CB_OpType_t op_type, size_t algo_idx, MPI_Request *req, CB_OperationData_t *data) {
    data->req = *req;
    data->rank = rank;
    data->peer = peer;
    data->op_type = op_type;
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

    MPI_CHECK(MPI_Wait(&data->req, MPI_STATUS_IGNORE), cleanup);

    data->t_end_ns = getCurrentTimeNS();
    data->req = MPI_REQUEST_NULL;

    cleanup:
        return err;
}

// NOTE: Double ptr as the data buff is owned by the list, so makes it easier to rearrange stuff
CB_Error_t CB_op_waitall(CB_OperationData_t ** const buff, size_t buff_len) {
    CB_Error_t err = CB_SUCCESS;
    if (!buff) {
        return CB_ERR_NULLPTR;
    }

    MPI_Request *reqs = NULL;
    CB_MALLOC(reqs, buff_len * sizeof(MPI_Request), cleanup);

    for (size_t i = 0; i < buff_len; i++) {
        buff[i]->t_wait_ns = getCurrentTimeNS();
        reqs[i] = buff[i]->req;
    }

    // FIXME: This sets the end time of all reqs to max time
    MPI_CHECK(MPI_Waitall(buff_len, reqs, MPI_STATUSES_IGNORE), cleanup);
    for(size_t i = 0; i < buff_len; i++) {
        buff[i]->req = MPI_REQUEST_NULL;
        buff[i]->t_end_ns = getCurrentTimeNS();
    }

    // TODO: Implement something like this (unsafe as waitany with null reqs is undefined behaviour)
    // for (size_t completed = 0; completed < buff_len; completed++) {
    //     int idx;
    //     MPI_CHECK(MPI_Waitany(buff_len, reqs, &idx, MPI_STATUS_IGNORE), cleanup);
    //     if (idx == MPI_UNDEFINED) break;
    //     buff[idx]->t_end_ns = getCurrentTimeNS();
    // }

    cleanup:
        free(reqs);
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

    static_assert(sizeof(MPI_Request) == sizeof(uint64_t), "MPI_Request size mismatch");

    static const struct {
        MPI_Aint     offset;
        MPI_Datatype type;
    } fields[] = {
        // FIXME: works only on 64-bit systems
        { offsetof(CB_OperationData_t, req),        MPI_UINT64_T }, /* MPI_Request as 8-byte ptr value */
        { offsetof(CB_OperationData_t, rank),        MPI_INT },
        { offsetof(CB_OperationData_t, peer),        MPI_INT },
        { offsetof(CB_OperationData_t, op_type),        MPI_INT },
        { offsetof(CB_OperationData_t, algo_idx),   MPI_UINT64_T },
        { offsetof(CB_OperationData_t, t_start_ns), MPI_UINT64_T },
        { offsetof(CB_OperationData_t, t_wait_ns),  MPI_UINT64_T },
        { offsetof(CB_OperationData_t, t_end_ns),   MPI_UINT64_T },
    };

    // NOTE: This computes nfields at compile time (equivalent to templatge heavy programming in c++)
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
    printf("Operation  : %s\n", CB_optype_str(data->op_type));
    printf("Rank  : %d\n", data->rank);
    printf("Peer  : %d\n", data->peer);
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
