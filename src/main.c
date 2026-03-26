#include "CollBench/bench.h"
#include "CollBench/dist_list.h"
#include "CollBench/errors.h"
#include "CollBench/export.h"
#include "CollBench/init.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>

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

int main(int argc, char **argv) {
    CB_Error_t err = CB_SUCCESS;
    MPI_Init(&argc, &argv);
    CB_init();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buff[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    MPI_Request req;
    CB_DistList_t *list;
    CB_dlist_init(&list, CB_DEF_INIT_SIZE);

    int root  = 0;
    int s     = ceil(log2(size));
    int nb_r  = rank2nb(rank, s);
    int mask  = (1 << s) - 1;
    int rcvd  = (root == rank);
    int round = 0;

    while (mask > 0) {
        int nb_q = nb_r ^ mask;
        int q    = nb2rank(nb_q, s);

        if (q >= size) {
            mask >>= 1;
            round++;
            continue;
        }

        if (rcvd) {
            CB_OP_LWRAP_NONBLOCKING(list, rank, CB_OP_SEND, round,
                MPI_Isend(buff, 10, MPI_INT, q, 0, MPI_COMM_WORLD, &req),
                req, cleanup);
        } else {
            int eq_lsb = nb_r & mask;
            if (eq_lsb == 0 || eq_lsb == mask) {
                CB_OP_LWRAP_NONBLOCKING(list, rank, CB_OP_RECV, round,
                    MPI_Irecv(buff, 10, MPI_INT, q, 0, MPI_COMM_WORLD, &req),
                req, cleanup);
                rcvd = 1;
            }
        }

        mask >>= 1;
        round++;
    }

    for(int i = 0; i < 10; i++) {
        if(buff[i] != i + 1) {
            MPI_Abort(MPI_COMM_WORLD, 10);
        }
    }

    CB_DistList_t *full_list = NULL;
    CB_dlist_gather(list, MPI_COMM_WORLD, 0, &full_list);

    if (rank == 0) {
        CB_dlist_pprint(full_list);
        CB_dlist_export_json(full_list, "out.json");
    }

cleanup:
    CB_dlist_free(list);
    CB_dlist_free(full_list);
    CB_finalize();
    MPI_Finalize();
    return err;
}
