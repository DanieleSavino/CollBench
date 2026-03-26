#include "CollBench/dist_list.h"
#include "CollBench/bench.h"
#include "CollBench/errors.h"
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CB_Error_t CB_dlist_init(CB_DistList_t **list, size_t init_size) {
    CB_Error_t err = CB_SUCCESS;

    CB_DistList_t *der_list = NULL;
    CB_MALLOC(der_list, sizeof(CB_DistList_t), cleanup);

    CB_MALLOC(der_list->_buffer, init_size * sizeof(CB_OperationData_t), cleanup);

    der_list->_buff_size = init_size;
    der_list->len = 0;
    *list = der_list;

    return CB_SUCCESS;

    cleanup:
        free(der_list);
        return err;
}

CB_Error_t CB_dlist_push(CB_DistList_t * const list, MPI_Request *req, int rank, CB_OpType_t op_type, size_t algo_idx, CB_OperationData_t **out) {
    size_t buff_size = list->_buff_size;
    size_t list_len = list->len;

    if(list_len >= buff_size) {
        void *tmp = realloc(list->_buffer, list->_buff_size * 2 * sizeof(CB_OperationData_t));
        if(!tmp) {
            return CB_ERR_OUT_OF_MEM;
        }
        list->_buffer = tmp;
        list->_buff_size *= 2;
    }

    CB_op_init_ext(req, rank, op_type, algo_idx, &list->_buffer[list_len]);
    *out = &list->_buffer[list_len];
    list->len++;

    return CB_SUCCESS;
}

CB_Error_t CB_dlist_get(const CB_DistList_t *const list, size_t idx, CB_OperationData_t **out) {
    CB_OperationData_t *buffer = list->_buffer;
    size_t list_len = list->len;

    if(idx >= list_len) {
        return CB_ERR_OUT_OF_BOUNDS;
    }

    *out = &(buffer[idx]);
    return CB_SUCCESS;
}

CB_Error_t CB_dlist_pop(CB_DistList_t * const list, CB_OperationData_t **out) {
    CB_Error_t err = CB_SUCCESS;

    CB_OperationData_t *buffer = list->_buffer;
    size_t list_len = list->len;

    if(list_len == 0) {
        return CB_ERR_OUT_OF_BOUNDS;
    }

    if(out != NULL) {
        CB_OperationData_t *data = NULL;
        CB_MALLOC(data, sizeof(CB_OperationData_t), cleanup);
        if(!data) {
            return CB_ERR_OUT_OF_MEM;
        }

        memcpy(data, &(buffer[list_len - 1]), sizeof(CB_OperationData_t));
        *out = data;
    }

    list->len--;

    cleanup:
        return err;
}

CB_Error_t CB_dlist_free(CB_DistList_t * const list) {
    if(list == NULL)
        return CB_ERR_NULLPTR;

    free(list->_buffer);
    free(list);

    return CB_SUCCESS;
}

CB_Error_t CB_dlist_pprint(const CB_DistList_t * const list) {
    if (!list) {
        fprintf(stderr, "CB_DistList_t: (null)\n");
        return CB_ERR_NULLPTR;
    }
    printf("CB_DistList_t {\n");
    printf("  len       = %zu\n", list->len);
    printf("  buff_size = %zu\n", list->_buff_size);
    printf("  buffer    = %p\n", (void *)list->_buffer);
    printf("}\n");
    for (size_t i = 0; i < list->len; i++) {
        printf("[%zu] ", i);
        CB_Error_t err = CB_op_pprint(&list->_buffer[i]);
        if (err != CB_SUCCESS) return err;
    }
    return CB_SUCCESS;
}

static inline CB_Error_t CB_dlist_gather_meta(const CB_DistList_t * const list, MPI_Comm comm, int root, int **counts, int **displs, size_t *out_len) {
    CB_Error_t err = CB_SUCCESS;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    *counts = NULL;
    *displs = NULL;
    *out_len = 0;

    if(rank != root) {
        MPI_Gather(&(list->len), 1, MPI_UINT64_T, NULL, 0, MPI_UINT64_T, root, comm);
        return CB_SUCCESS;
    }

    size_t *counts_buff = NULL;
    int *int_counts_buff = NULL, *displs_buff = NULL;
    CB_MALLOC(counts_buff, size * sizeof(size_t), cleanup);
    CB_MALLOC(int_counts_buff, size * sizeof(int), cleanup);
    CB_MALLOC(displs_buff, size * sizeof(int), cleanup);

    MPI_CHECK(
        MPI_Gather(
            &(list->len),
            1,
            MPI_UINT64_T,
            counts_buff,
            1,
            MPI_UINT64_T,
            root,
            comm
        ),
        cleanup
    );

    size_t offset = 0;
    for(int i = 0; i < size; i++) {
        displs_buff[i] = (int)offset;
        int_counts_buff[i] = (int)counts_buff[i];
        offset += counts_buff[i];
    }

    free(counts_buff);

    CB_INT_OF_CHECK(offset, cleanup);

    *counts = int_counts_buff;
    *displs = displs_buff;
    *out_len = offset;

    return CB_SUCCESS;

    cleanup:
        free(int_counts_buff);
        free(displs_buff);
        free(counts_buff);

        return err;
}

CB_Error_t CB_dlist_gather(const CB_DistList_t * const list, MPI_Comm comm, int root, CB_DistList_t **out) {
    CB_Error_t err = CB_SUCCESS;
    *out = NULL;

    int rank, *counts = NULL, *displs = NULL;
    CB_DistList_t *out_list = NULL;

    MPI_CHECK(MPI_Comm_rank(comm, &rank), cleanup);

    size_t out_len;
    CB_CHECK(CB_dlist_gather_meta(list, comm, root, &counts, &displs, &out_len), cleanup);

    if(rank == root)
        CB_CHECK(CB_dlist_init(&out_list, out_len), cleanup);
    else
        out_list = NULL;

    CB_INT_OF_CHECK(list->len, cleanup);

    MPI_CHECK(
        MPI_Gatherv(
            list->_buffer,
            (int)list->len,
            CB_OP_DATATYPE,
            rank == root ? out_list->_buffer : NULL,
            counts,
            displs,
            CB_OP_DATATYPE,
            root,
            comm
        ),
        cleanup
    );

    free(counts);
    free(displs);

    if(rank == root) {
        out_list->len = out_len;
        *out = out_list;
    }

    return CB_SUCCESS;

    cleanup:
        free(counts);
        free(displs);
        CB_dlist_free(out_list);
        return err;
}
