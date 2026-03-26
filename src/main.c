#include "CollBench/dist_list.h"
#include "CollBench/errors.h"
#include "CollBench/init.h"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    CB_Error_t err = CB_SUCCESS;

    MPI_Init(&argc, &argv);
    CB_init();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    MPI_Request req;
    CB_DistList_t *list;
    CB_dlist_init(&list, CB_DEF_INIT_SIZE);

    if(rank == 0) {
        CB_OP_LWRAP_NONBLOCKING(list, 0, MPI_Isend(buff, 10, MPI_INT, 1, 0, MPI_COMM_WORLD, &req), req, cleanup);
    }
    else {
        CB_OP_LWRAP_NONBLOCKING(list, 0, MPI_Irecv(buff, 10, MPI_INT, 0, 0, MPI_COMM_WORLD, &req), req, cleanup);
    }

    CB_DistList_t *full_list = NULL;
    CB_dlist_gather(list, MPI_COMM_WORLD, 0, &full_list);

    if(rank == 0)
        CB_dlist_pprint(full_list);

    cleanup:
        CB_dlist_free(list);
        CB_dlist_free(full_list);

        CB_finalize();
        MPI_Finalize();
}
