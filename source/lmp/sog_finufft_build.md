# SOG FINUFFT Build and Runtime Notes

This file documents how to build FINUFFT and wire it into
`kspace_style sog` without relying on implicit `libfinufft.so` search.

## 1) Build FINUFFT (from upstream GitHub)

Repository:
https://github.com/flatironinstitute/finufft.git

Minimal shared-library build:

1. `git clone https://github.com/flatironinstitute/finufft.git`
2. `cd finufft`
3. `mkdir -p build && cd build`
4. `cmake .. -DBUILD_SHARED_LIBS=ON -DFINUFFT_STATIC_LINKING=OFF`
5. `cmake --build . -j`

Expected output library:

- `build/src/libfinufft.so`

Alternative helper script in this directory:

1. `cd source/lmp`
2. `./build_finufft.sh`

This script clones FINUFFT from upstream and builds `libfinufft.so`
with `BUILD_SHARED_LIBS=ON` and `FINUFFT_STATIC_LINKING=OFF`.

## 2) Tell SOG where FINUFFT is

The SOG FINUFFT backend now expects an explicit library path.
Use one of the following:

- LAMMPS command option:
  - `kspace_style sog ... use_finufft yes finufft_library /abs/path/to/libfinufft.so`
- Environment variable:
  - `DP_SOG_FINUFFT_LIBRARY=/abs/path/to/libfinufft.so`

If neither is provided, SOG prints a warning with build instructions and
falls back to PPPM.

When using package install script `source/lmp/Install.sh` in install mode,
it also prints a reminder to run `./build_finufft.sh`.

## 3) Parallel support status

Current implementation status in `source/lmp/sog.cpp`:

- MPI domain decomposition: not supported for the FINUFFT path.
  - Condition: `comm->nprocs` must be 1.
  - Otherwise it falls back to PPPM.
- OpenMP threading inside one MPI rank: supported by FINUFFT itself.
  - Control with `OMP_NUM_THREADS` (and FINUFFT/OpenMP settings).
- Triclinic box on FINUFFT path: not supported currently.
  - Orthorhombic periodic box is required for FINUFFT path.

In short: current FINUFFT backend is single-rank (MPI=1) + threaded (OpenMP)
inside that rank.
