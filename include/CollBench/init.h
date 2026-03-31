#pragma once

#include "CollBench/errors.h"
#include "CollBench/dist_list.h"
#include "CollBench/export.h"

#ifdef CB_PROFILE

static inline CB_Error_t CB_init(void) {
    CB_Error_t err = CB_SUCCESS;
    CB_CHECK(CB_op_datatype_init(), cleanup);

    cleanup:
        return err;
}

static inline CB_Error_t CB_finalize(void) {
    CB_Error_t err = CB_SUCCESS;
    CB_CHECK(CB_op_datatype_free(), cleanup);

    cleanup:
        return err;
}

#define CB_COLL_START() \
    CB_Error_t err = CB_SUCCESS; \
    CB_DistList_t *_list = NULL, *_full_list = NULL; \
    CB_CHECK(CB_dlist_init(&_list, CB_DEF_INIT_SIZE), _CB_cleanup_label);

#define CB_COLL_END(comm, rank, root, out_path) \
        CB_CHECK(CB_dlist_gather(_list, comm, root, &_full_list), _CB_gather_failed); \
    _CB_gather_failed: \
        CB_CHECK(CB_dlist_free(_list), _CB_cleanup_label); \
        if(rank == root) \
            CB_CHECK(CB_dlist_export_json(_full_list, out_path), _CB_cleanup_label); \
    _CB_cleanup_label: \
        CB_dlist_free(_full_list)

#else

static inline CB_Error_t CB_init(void) {
    return CB_SUCCESS;
}

static inline CB_Error_t CB_finalize(void) {
    return CB_SUCCESS;
}

#define CB_COLL_START() ;
#define CB_COLL_END(comm, rank, root, out_path) ;

#endif
