# CollBench

> [!WARNING]
> This README was AI-generated from the source code and headers. Some details may be imprecise, incomplete, or outdated. When in doubt, refer to the headers in `include/CollBench/` as the authoritative reference.

**CollBench** is a lightweight C library for instrumenting and benchmarking MPI collective communication algorithms. Developed as part of a Bachelor's thesis in Computer Engineering at **Sapienza Università di Roma**, it provides a clean API for timing individual MPI operations, aggregating results across processes, and exporting structured profiling data for analysis.

---

## Overview

Implementing custom collective algorithms in MPI is straightforward — measuring them precisely is not. CollBench fills that gap: replace your point-to-point calls with `CB_LSEND` / `CB_LRECV`, bracket the function with `CB_COLL_START` / `CB_COLL_END`, and get per-operation nanosecond-resolution timing exported as JSON across all processes. Compile without `-DCB_PROFILE` and the library adds no overhead whatsoever.

---

## Features

- **Nanosecond-resolution timing** per send/recv, with separate start, wait, and end timestamps
- **Zero-overhead toggle**: compile without `-DCB_PROFILE` and all instrumentation disappears, leaving plain blocking MPI calls
- **Automatic aggregation**: per-process records are gathered to a root process and exported as JSON

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
└── lib/libcollbench_debug.a
```

---

## Project Structure

```
CollBench/
├── include/CollBench/
│   ├── bench.h        # Core operation types and timing API
│   ├── dist_list.h    # Distributed operation list
│   ├── errors.h       # Error codes and utility macros
│   ├── export.h       # JSON export
│   └── init.h         # Top-level init/finalize and collective wrappers
├── src/
│   ├── bench.c
│   ├── dist_list.c
│   └── export.c
├── demos/
│   └── bine_bcast/    # Example: BiNE broadcast implementation
├── scripts/
│   └── tree.py
└── Makefile
```

---

## API

Include a single header and call `CB_init` / `CB_finalize` around `MPI_Init` / `MPI_Finalize`:

```c
#include "CollBench/init.h"

CB_init();
// ...
CB_finalize();
```

Inside a collective function, open and close an instrumented region with:

```c
CB_COLL_START()   // top of the function — allocates per-process record list
CB_COLL_END(comm, root, "output.json")   // bottom — gathers records and writes JSON on root
```

Replace all point-to-point calls inside the collective with:

```c
CB_LSEND(rank, algo_idx, buff, count, datatype, dest,   tag, comm)
CB_LRECV(rank, algo_idx, buff, count, datatype, source, tag, comm)
```

`algo_idx` is an arbitrary integer you assign to tag which algorithm step the operation belongs to — it appears in the JSON output for later analysis.

When `CB_PROFILE` is **not** defined at compile time, all four macros compile away to plain blocking send/recv calls with zero overhead.

---

## Usage Example

Below is an abbreviated version of the BiNE broadcast demo (`demos/bine_bcast/main.c`), which implements a distance-halving / level-doubling broadcast and instruments it with CollBench:

```c
#include "CollBench/init.h"
#include <mpi.h>

int bine_bcast_dhlv(void *buffer, int count, MPI_Datatype datatype,
                    int root, MPI_Comm comm)
{
    CB_COLL_START();  // declares _list, allocates storage

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* ... algorithm logic using CB_LSEND / CB_LRECV ... */

    CB_COLL_END(comm, root, "bine_bcast.json");
    return MPI_SUCCESS;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    CB_init();

    int *buff = malloc(sizeof(int) * 100);
    /* fill buff on rank 0 */

    bine_bcast_dhlv(buff, 100, MPI_INT, 0, MPI_COMM_WORLD);

    CB_finalize();
    MPI_Finalize();
}
```

Compile with `-DCB_PROFILE` to enable profiling:

```sh
mpicc -DCB_PROFILE -ICollBench/include main.c -lcollbench -lm -o demo
mpirun -n 8 ./demo
# produces bine_bcast.json with per-operation timing for all 8 processes
```

Omit `-DCB_PROFILE` for a production build where all instrumentation is compiled away.

---

## Contributing

This is a personal research library developed for a Bachelor's thesis — it exists because it's good enough for the job, not as a general-purpose tool. It is not actively maintained or intended for external use.

That said, if you stumble across it and find a bug or have a suggestion, feel free to open an issue.

---

## License

MIT License

Copyright (c) 2024 Daniele — Sapienza Università di Roma

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
