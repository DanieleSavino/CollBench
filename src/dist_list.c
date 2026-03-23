#include "CollBench/dist_list.h"
#include "CollBench/bench.h"
#include "CollBench/errors.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CB_Error_t CB_dlist_init(CB_DistList_t **list, size_t init_size) {
    CB_DistList_t *der_list = malloc(sizeof(CB_DistList_t));
    if(!der_list) goto oom_list;

    der_list->_buffer = malloc(init_size * sizeof(CB_OperationData_t));
    if(!der_list->_buffer) goto oom_buff;

    der_list->_buff_size = init_size;
    der_list->len = 0;
    *list = der_list;

    return CB_SUCCESS;

    oom_buff:
        free(der_list);
    oom_list:
        return CB_ERR_OUT_OF_MEM;
}

CB_Error_t CB_dlist_push(CB_DistList_t * const list, MPI_Request *req, size_t algo_idx, CB_OperationData_t **out) {
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

    CB_op_init_ext(req, algo_idx, &list->_buffer[list_len]);
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
    CB_OperationData_t *buffer = list->_buffer;
    size_t list_len = list->len;

    if(list_len == 0) {
        return CB_ERR_OUT_OF_BOUNDS;
    }

    if(out != NULL) {
        CB_OperationData_t *data = malloc(sizeof(CB_OperationData_t));
        if(!data) {
            return CB_ERR_OUT_OF_MEM;
        }

        memcpy(data, &(buffer[list_len - 1]), sizeof(CB_OperationData_t));
        *out = data;
    }

    list->len--;

    return CB_SUCCESS;
}

CB_Error_t CB_dlist_free(CB_DistList_t * const list) {
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
