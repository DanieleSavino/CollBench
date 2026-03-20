#include "bench.h"
#include "errors.h"
#include <mpi.h>

int main(void) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    MPI_Request req;
    CB_OperationData_t *data;

    if(rank == 0)
        CB_OP_WRAP_NONBLOCKING("Send from root", 0, MPI_Isend(buff, 10, MPI_INT, 1, 0, MPI_COMM_WORLD, &req), &data, req);
    else 
        CB_OP_WRAP_NONBLOCKING("Recv from 1", 1, MPI_Irecv(buff, 10, MPI_INT, 0, 0, MPI_COMM_WORLD, &req), &data, req);

    CB_op_pprint(data);

    MPI_Finalize();


}
