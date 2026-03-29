#include "CollBench/dist_list.h"
#include "CollBench/errors.h"
#include "CollBench/init.h"
#include <math.h>
#include <mpi.h>

int rank2nb(int32_t rank, int bits) {
    const int size = (1 << bits);
    if(rank > 0x55555555) return -1;

    const uint32_t mask = 0xAAAAAAAA;
    const int32_t val = (mask + rank) ^ mask;

    return val & (size - 1);
}

int nb2rank(int32_t nb, int bits) {
    const int size = (1 << bits);
    const uint32_t mask = 0xAAAAAAAA;
    const int32_t val = (mask ^ nb) - mask;
    return val & (size - 1);
}

int mod(int a, int b){
    int r = a % b;
    return r < 0 ? r + b : r;
}

int bine_bcast_dhlv(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    CB_COLL_START();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int s = ceil(log2(size));

    int vrank = mod(rank - root, size);
    int nb_r = rank2nb(vrank, s);

    int mask = (1 << s) - 1;
    int rcvd = root == rank;
    while(mask > 0) {

        int nb_q = nb_r ^ mask;
        int vq = nb2rank(nb_q, s);
        int q = mod(vq + root, size);

        if(q >= size) {
            mask >>= 1;
            continue;
        }

        if(rcvd) {
            CB_LSEND(rank, s - __builtin_popcount(mask), buffer, count, datatype, q, 0, comm);
        }
        else {
            int eq_lsb = nb_r & mask;
            if(eq_lsb == 0 || eq_lsb == mask) {
                CB_LRECV(rank, s - __builtin_popcount(mask), buffer, count, datatype, q, 0, comm);
                rcvd = 1;
            }
        }

        mask >>= 1;
    }

    CB_COLL_END(comm, root, "bine_bcast.json");

    return MPI_SUCCESS;
}

int main(int argc, char **argv) {
    CB_Error_t err = CB_SUCCESS;
    MPI_Init(&argc, &argv);
    CB_init();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *buff;
    CB_MALLOC(buff, sizeof(int) * 10, cleanup);
    for(int i = 0; i < 10; i++) {
        buff[i] = rank == 0 ? i : 0;
    }

    bine_bcast_dhlv(buff, 10, MPI_INT, 0, MPI_COMM_WORLD);

    for(int i = 0; i < 10; i++) {
        if(buff[i] != i) {
            MPI_Abort(MPI_COMM_WORLD, 10);
        }
    }

    cleanup:
        CB_finalize();
        MPI_Finalize();
        return err;
}
