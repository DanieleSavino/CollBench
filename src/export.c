#include "CollBench/export.h"
#include "CollBench/bench.h"
#include "CollBench/errors.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <string.h>
#include <libgen.h>

static CB_Error_t CB_mkdir_p(const char *path) {
    char tmp[256];
    strncpy(tmp, path, sizeof(tmp) - 1);
    char *dir = dirname(tmp);
    if (strcmp(dir, ".") == 0 || strcmp(dir, "/") == 0) return 0;
    char buf[256];
    strncpy(buf, dir, sizeof(buf) - 1);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0755);
            *p = '/';
        }
    }

    return mkdir(buf, 0755) ? CB_ERR_IO : CB_SUCCESS;
}

CB_Error_t CB_dlist_export_json(const CB_DistList_t *list, const char *path) {
    CB_Error_t err = CB_SUCCESS;

    if (!list || !path) return CB_ERR_INVALID_ARG;
    CB_CHECK(CB_mkdir_p(path), ret);

    FILE *f = fopen(path, "w");
    if (!f) return CB_ERR_IO;

    fprintf(f, "{\n");
    fprintf(f, "  \"count\": %zu,\n", list->len);
    fprintf(f, "  \"operations\": [\n");

    for (size_t i = 0; i < list->len; i++) {
        const CB_OperationData_t *op = &list->_buffer[i];

        uint64_t t_issue_ns  = op->t_wait_ns  - op->t_start_ns; /* time from start to MPI_Wait */
        uint64_t t_total_ns  = op->t_end_ns   - op->t_start_ns;

        fprintf(f,
            "    {\n"
            "      \"index\":       %zu,\n"
            "      \"operation_type\":        \"%s\",\n"
            "      \"rank\":        %d,\n"
            "      \"peer\":        %d,\n"
            "      \"algo_idx\":    %zu,\n"
            "      \"t_start_ns\":  %"PRIu64",\n"
            "      \"t_wait_ns\":   %"PRIu64",\n"
            "      \"t_end_ns\":    %"PRIu64",\n"
            "      \"t_issue_ns\":  %"PRIu64",\n"
            "      \"t_total_ns\":  %"PRIu64"\n"
            "    }%s\n",
            i,
            CB_optype_str(op->op_type),
            op->rank,
            op->peer,
            op->algo_idx,
            op->t_start_ns,
            op->t_wait_ns,
            op->t_end_ns,
            t_issue_ns,
            t_total_ns,
            i + 1 < list->len ? "," : ""
        );
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);

    ret:
        return err;
}
