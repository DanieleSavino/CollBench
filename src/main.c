#include "CollBench/bench.h"
#include "CollBench/dist_list.h"
#include <mpi.h>

int main(void) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    MPI_Request req;
    CB_DistList_t *list;
    CB_dlist_init(&list, 1);

    if(rank == 0)
        CB_OP_LWRAP_NONBLOCKING(list, 0, MPI_Isend(buff, 10, MPI_INT, 1, 0, MPI_COMM_WORLD, &req), req);
    else 
        CB_OP_LWRAP_NONBLOCKING(list, 1, MPI_Irecv(buff, 10, MPI_INT, 0, 0, MPI_COMM_WORLD, &req), req);

    CB_dlist_pprint(list);
    CB_dlist_free(list);


    MPI_Finalize();
}
