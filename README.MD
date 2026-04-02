# CollBench

> [!WARNING]
> This README was written from the source code and headers. When in doubt, refer to the headers in `include/CollBench/` as the authoritative reference.

**CollBench** is a lightweight C library for instrumenting and benchmarking MPI collective communication algorithms. Developed as part of a Bachelor's thesis in Computer Engineering at **Sapienza Università di Roma**, it provides a clean API for timing individual MPI operations, aggregating results across processes, and exporting structured profiling data for analysis.

---

## Overview

Implementing custom collective algorithms in MPI is straightforward — measuring them precisely is not. CollBench fills that gap: replace your point-to-point calls with `CB_LSEND` / `CB_LRECV`, bracket the function with `CB_COLL_START` / `CB_COLL_END`, and get per-operation nanosecond-resolution timing exported as JSON across all processes. Compile without `-DCB_PROFILE` and the library adds zero overhead.

---

## Features

- **Nanosecond-resolution timing** per send/recv, with separate start, wait, and end timestamps (`t_start_ns`, `t_wait_ns`, `t_end_ns`)
- **Zero-overhead toggle**: compile without `-DCB_PROFILE` and all instrumentation disappears, leaving plain MPI calls
- **Blocking and nonblocking variants**: `CB_LSEND`/`CB_LRECV` for logically blocking sends/recvs; `CB_ILSEND`/`CB_ILRECV` + `CB_LWAIT`/`CB_LWAITALL` for fully nonblocking workflows
- **Automatic aggregation**: per-process records are gathered to a root process via `MPI_Gatherv` and exported as JSON

---

## Requirements

- An MPI implementation (e.g. Open MPI, MPICH)
- C99-compatible compiler accessible as `mpicc`
- GNU Make
- *(Optional)* [`bear`](https://github.com/rizsotto/Bear) for generating `compile_commands.json` for LSP support

---

## Building

```sh
# Release build (default)
make

# Debug build
make debug

# Generate compile_commands.json (requires bear)
make compile_commands.json

# Clean all build artifacts
make clean
```

Build outputs are placed under `build/`:

```
build/
├── lib/libcollbench.a
└── obj/
    ├── bench.o
    ├── dist_list.o
    └── export.o
```

---

## Project Structure

```
CollBench/
├── include/CollBench/
│   ├── bench.h        # Core operation record (CB_OperationData_t), timing API, standalone macros
│   ├── dist_list.h    # Distributed operation list (CB_DistList_t) and list-based profiling macros
│   ├── errors.h       # Error codes (CB_Error_t) and utility macros
│   ├── export.h       # JSON export (CB_dlist_export_json)
│   └── init.h         # Top-level init/finalize and CB_COLL_START/CB_COLL_END macros
├── src/
│   ├── bench.c
│   ├── dist_list.c
│   └── export.c
├── demos/
│   ├── bine_bcast/
│   │   ├── main.c
│   │   └── Makefile -> ../shared_Makefile
│   └── shared_Makefile
├── scripts/
│   ├── butterfly.py
│   └── tree.py
├── requirements.txt
└── Makefile
```

---

## API

### Initialization and finalization

Include the top-level header and call `CB_init` / `CB_finalize` between `MPI_Init` and `MPI_Finalize`. `CB_init` registers the `CB_OperationData_t` MPI derived datatype; `CB_finalize` frees it. Both are no-ops when `CB_PROFILE` is not defined.

```c
#include "CollBench/init.h"

MPI_Init(&argc, &argv);
CB_init();

// ... collective calls ...

CB_finalize();
MPI_Finalize();
```

### Instrumenting a collective

Open and close a profiling scope at the top and bottom of a collective function with `CB_COLL_START` / `CB_COLL_END`. `CB_COLL_START` declares the local `err`, `_list`, and `_full_list` variables and initializes the per-process record list. `CB_COLL_END` gathers records from all ranks onto root, exports them to JSON, and frees both lists.

```c
int my_bcast(void *buffer, int count, MPI_Datatype datatype,
             int root, MPI_Comm comm)
{
    CB_COLL_START();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* ... algorithm using CB_LSEND / CB_LRECV / CB_ILSEND / CB_ILRECV ... */

    CB_COLL_END(comm, rank, root, "output.json");
    return MPI_SUCCESS;
}
```

Mismatching `CB_COLL_START` and `CB_COLL_END` is caught at compile time when `CB_PROFILE` is defined.

### Point-to-point macros

All macros below record into the implicit `_list` declared by `CB_COLL_START`. They are no-ops (plain MPI calls) when `CB_PROFILE` is not defined.

#### Blocking

Post an `MPI_Isend` / `MPI_Irecv` and immediately wait, recording `t_start_ns`, `t_wait_ns`, and `t_end_ns`.

```c
CB_LSEND(rank, algo_idx, buff, count, datatype, dest,   tag, comm)
CB_LRECV(rank, algo_idx, buff, count, datatype, source, tag, comm)
```

#### Nonblocking

Post an `MPI_Isend` / `MPI_Irecv` and return immediately.

```c
CB_ILSEND(rank, algo_idx, buff, count, datatype, dest,   tag, comm, req_ref)
CB_ILRECV(rank, algo_idx, buff, count, datatype, source, tag, comm, req_ref)
```

> [!WARNING]
> After posting, the value written into `*req_ref` is **overwritten with an opaque internal list index** — it is no longer a valid `MPI_Request`. Never pass it to `MPI_Wait`, `MPI_Waitall`, or any other MPI function. The real MPI handle is preserved inside the operation record (`data->req`) and is used transparently by `CB_LWAIT` / `CB_LWAITALL`. Bypassing these macros will corrupt the CollBench data structure and leave operations without completion timestamps.

Wait on a single previously posted nonblocking operation:

```c
CB_LWAIT(req_ref)
```

Wait on all requests in an array, recording a shared `t_end_ns` across all of them (equivalent to `MPI_Waitall`):

```c
CB_LWAITALL(reqs, buff_len)
```

`algo_idx` is an arbitrary integer you assign to identify which step of the algorithm an operation belongs to — it appears in the JSON output for later analysis.

---

## Usage Examples

### Blocking sends and recvs (broadcast-style)

For tree-shaped collectives where each rank either sends or receives at each step, use `CB_LSEND` / `CB_LRECV`.

```c
int my_bcast(void *buff, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    CB_COLL_START();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int step = 0;
    while (/* rounds remain */) {
        int peer = /* compute peer */;

        if (has_data) {
            CB_LSEND(rank, step, buff, count, datatype, peer, 0, comm);
        } else if (/* should receive */) {
            CB_LRECV(rank, step, buff, count, datatype, peer, 0, comm);
        }

        step++;
    }

    CB_COLL_END(comm, rank, root, "out/bine_bcast.json");
    return MPI_SUCCESS;
}
```

### Nonblocking sends and recvs (allgather-style)

For algorithms that post multiple sends and recvs per step before waiting, use `CB_ILSEND` / `CB_ILRECV` with a shared `MPI_Request` array, then drain it with `CB_LWAITALL` at the end of each step.

`req_idx` is safe to use as the index expression — `req_ref` is captured into a local pointer inside the macro to avoid double-evaluation of expressions with side effects (e.g. `&reqs[req_idx++]`).

```c
int my_allgather(const void *sendbuff, int sendcount, MPI_Datatype sendtype,
                 void *recvbuff, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm) {
    CB_COLL_START();

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int req_idx = 0;
    MPI_Request *reqs = malloc(size * 2 * sizeof(MPI_Request));

    int step = 0;
    while (/* rounds remain */) {
        int peer = /* compute peer */;

        /* post all sends and recvs for this step */
        CB_ILSEND(rank, step, send_ptr, sendcount, sendtype, peer, 0, comm,
                  &reqs[req_idx++]);
        CB_ILRECV(rank, step, recv_ptr, recvcount, recvtype, peer, 0, comm,
                  &reqs[req_idx++]);

        /* wait for all posted operations before advancing */
        CB_LWAITALL(reqs, req_idx);
        req_idx = 0;
        step++;
    }

    free(reqs);
    CB_COLL_END(comm, rank, 0, "out/bine_allgather.json");
    return MPI_SUCCESS;
}
```

> [!NOTE]
> After `CB_ILSEND` / `CB_ILRECV` returns, `reqs[i]` holds an internal list index, not a real MPI request handle. The array is only valid as input to `CB_LWAIT` / `CB_LWAITALL` — do not pass its entries to any MPI function directly.

### Compiling

```sh
# With profiling
mpicc -DCB_PROFILE -ICollBench/include main.c -lcollbench -lm -o demo
mpirun -n 8 ./demo
# produces the JSON file(s) specified in CB_COLL_END

# Without profiling — zero overhead, plain MPI
mpicc -ICollBench/include main.c -lcollbench -lm -o demo
```

---

## Internal: request hijacking

When profiling is enabled, `CB_ILSEND` and `CB_ILRECV` post the MPI operation and store the real `MPI_Request` handle inside the operation record. They then **overwrite** `*req_ref` with the record's index in `_list`, cast through `uintptr_t`:

```c
*req = (MPI_Request)(uintptr_t)(_list->len - 1);
```

`CB_LWAIT` and `CB_LWAITALL` reverse this by casting back to `size_t` and calling `CB_dlist_get` by index, then forwarding the real handle (stored in `data->req`) to `CB_op_wait` / `CB_op_waitall`, which call `MPI_Wait` / `MPI_Waitall` and record the completion timestamps.

This sidesteps the fragility of using `MPI_Request` values as lookup keys — MPI is free to reuse handle addresses after a request completes, which made the original `CB_dlist_getbyreq` approach unreliable across multi-step algorithms. `CB_dlist_getbyreq` is retained in the API but is no longer used internally.

The cast is safe on any platform where `sizeof(MPI_Request) >= sizeof(uintptr_t)`, which holds on all supported 32- and 64-bit targets. Overflow of `_list->len` past `UINTPTR_MAX` is not a practical concern.

---

## Visualization

Two Python scripts in `scripts/` consume the JSON output and produce figures. Install dependencies first:

```sh
pip install -r requirements.txt
```

### `scripts/tree.py` — broadcast / gather tree

Draws the communication tree for a single-root collective (broadcast or gather). Direction is auto-detected from the JSON but can be overridden.

```sh
# Basic usage
python scripts/tree.py out/bine_bcast_dhlv.json

# Custom root, high-DPI output
python scripts/tree.py --root 3 --dpi 300 --out tree.png out/bine_bcast_dhlv.json

# Gather direction, hide latency labels
python scripts/tree.py --direction gather --no-latency out/gather.json

# Custom round palette
python scripts/tree.py --palette "#E05C5C,#5B8FF9,#5AD8A6" out/bine_bcast_dhlv.json
```

| Option | Default | Description |
|---|---|---|
| `--root ROOT` | `0` | Root rank |
| `--direction {broadcast,gather}` | auto | Collective direction |
| `--dpi DPI` | `220` | Output DPI |
| `--out OUT` | `collbench_tree.png` | Output path |
| `--no-latency` | — | Hide per-edge latency labels |
| `--no-legend` | — | Hide the round legend |
| `--no-row-labels` | — | Hide left-side round labels |
| `--palette PALETTE` | built-in | Comma-separated hex colors for rounds |

### `scripts/butterfly.py` — allgather / multi-source visualizer

Supports three layout modes for collectives where every rank is both a sender and a receiver.

```sh
# Grid of broadcast trees, one per source rank (default)
python scripts/butterfly.py out/bine_allgather_b2b.json

# All trees overlaid on one canvas
python scripts/butterfly.py --mode overlay out/bine_allgather_b2b.json

# Rank-vs-step matrix with peer arrows
python scripts/butterfly.py --mode butterfly --dpi 300 --out fig.png out/bine_allgather_b2b.json

# Grid with fixed column count
python scripts/butterfly.py --mode grid --grid-cols 4 out/bine_allgather_b2b.json
```

| Option | Default | Description |
|---|---|---|
| `--mode {grid,overlay,butterfly}` | `grid` | Layout mode |
| `--dpi DPI` | auto | Output DPI |
| `--out OUT` | `collbench_<mode>.png` | Output path |
| `--grid-cols N` | auto | Number of columns in grid mode |
| `--no-latency` | — | Hide latency labels |
| `--no-legend` | — | Hide legend |
| `--palette PALETTE` | built-in | Comma-separated hex colors |

Both scripts support `--no-save` and `--no-show` to control whether the figure is written to disk or displayed interactively.

---

## Error Handling

All library functions return `CB_Error_t`. The macros `CB_CHECK` and `MPI_CHECK` (defined in `errors.h`) expect a local `CB_Error_t err` variable and a cleanup `goto` label in scope — both are provided automatically inside a `CB_COLL_START` / `CB_COLL_END` block.

| Code | Meaning |
|---|---|
| `CB_SUCCESS` | Success (guaranteed 0) |
| `CB_ERR_OUT_OF_MEM` | `malloc`/`realloc` returned NULL |
| `CB_ERR_OUT_OF_BOUNDS` | Index exceeded container size |
| `CB_ERR_NULLPTR` | A required pointer argument was NULL |
| `CB_ERR_INT_OF` | Integer overflow detected |
| `CB_ERR_IO` | I/O operation failed |
| `CB_ERR_INVALID_ARG` | Argument value is invalid or not found |
| `CB_ERR_MPI` | An MPI call returned a non-`MPI_SUCCESS` code |

---

## Contributing

This is a personal research library developed for a Bachelor's thesis — it exists because it's good enough for the job, not as a general-purpose tool. It is not actively maintained or intended for external use.

That said, if you stumble across it and find a bug or have a suggestion, feel free to open an issue.

---

## License

MIT License

Copyright (c) 2026 Daniele — Sapienza Università di Roma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
