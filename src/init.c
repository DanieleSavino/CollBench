#include "CollBench/init.h"
#include "CollBench/bench.h"
#include "CollBench/errors.h"

CB_Error_t CB_init(void) {
    CB_Error_t err = CB_SUCCESS;
    CB_CHECK(CB_op_datatype_init(), cleanup);

    cleanup:
        return err;
}

CB_Error_t CB_finalize(void) {
    CB_Error_t err = CB_SUCCESS;
    CB_CHECK(CB_op_datatype_free(), cleanup);

    cleanup:
        return err;
}
