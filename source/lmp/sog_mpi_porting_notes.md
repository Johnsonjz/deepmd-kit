# SOG MPI Porting Notes (PPPM Reference)

This note records how to port SOG to multi-MPI by reusing native PPPM flow.

## Key idea

For multi-MPI support, keep PPPM communication/FFT pipeline unchanged.
Only replace SOG-specific parts:

- interpolation/spreading kernel behavior (if needed)
- SOG Green's function construction

Everything else should follow PPPM path.

## PPPM flow to reuse

Reference: LAMMPS `src/KSPACE/pppm.cpp` and `source/lmp/pppm_dplr.cpp`.

Main compute flow:

1. `particle_map()`
2. `make_rho()`
3. `gc->reverse_comm(... REVERSE_RHO ...)`
4. `brick2fft()`
5. `poisson()`
6. `gc->forward_comm(... FORWARD_IK/FORWARD_AD ...)`
7. `fieldforce()`
8. `MPI_Allreduce` for global energy/virial

This flow already supports domain decomposition and MPI collectives.

## Suggested migration plan

1. Keep SOG class derived from PPPM and preserve PPPM compute pipeline.
2. Override SOG Green's function path first:
   - `compute_gf_ik()` / triclinic corresponding handling.
3. Keep force/virial accumulation rules aligned with PPPM conventions.
4. Validate in three steps:
   - MPI=1 parity with current validated SOG result.
   - MPI strong/weak scaling consistency on same trajectory/frame.
   - compare against existing PPPM/SOG single-rank baseline.

## Current FINUFFT backend status

Current FINUFFT path in `sog.cpp` is single-rank only (`comm->nprocs == 1`).
It is validated numerically for single-rank runs.

Before introducing multi-MPI FINUFFT or mixed strategy, keep this validated
path as a branch checkpoint.
