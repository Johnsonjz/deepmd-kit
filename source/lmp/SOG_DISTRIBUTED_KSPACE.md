# SOG distributed k-space implementation note

This is an internal implementation and benchmark note. It is not manuscript or SI material.

## Runtime backends

`kspace_style sog ... use_gpu yes gpu_mpi MODE` accepts three MPI policies:

- `replicated`: every rank gathers all atoms and executes the complete global CUDA mesh and cuFFT. This remains the legacy regression path.
- `root`: rank 0 executes one complete CUDA mesh and cuFFT, then scatters the force and charge-response-potential slices. It removes duplicate arithmetic but leaves the full k-space solve on the critical path.
- `distributed`: each rank spreads and gathers its local particles on its assigned CUDA device, while LAMMPS partitions the real-space mesh and reciprocal FFT across MPI ranks.

The distributed mode is enabled only for CubeS2 and QuadS windows and only when more than one MPI rank is present. A one-rank request falls back to the existing monolithic CUDA/cuFFT path.

## Distributed algorithm

The implementation follows the decomposition used by LAMMPS PPPM/GPU:

1. `sog_gpu_spread_brick` evaluates the local CubeS2-4/6 or QuadS-4/6 assignment on the GPU into an owned-plus-ghost scalar brick.
2. `Grid3d::reverse_comm` adds ghost-cell charge contributions into the owning ranks.
3. `Remap::perform` converts the owned brick layout to the distributed FFT pencil layout.
4. `FFT3d` performs the distributed forward transform.
5. Each rank evaluates only its local reciprocal modes, applies the SOG Form-B Green multiplier, and accumulates local energy, virial, diagonal-self, and self-virial sums.
6. One `MPI_Allreduce` combines the fourteen scalar/tensor accumulators. The charge-norm sum is reduced separately.
7. A second distributed `FFT3d` returns the energy potential directly to the owned brick layout.
8. `Grid3d::forward_comm` fills potential ghost cells.
9. `sog_gpu_gather_brick` evaluates the local potential and the analytic derivative of the same finite-grid spreading weights. This is the exact AD force of the discrete mesh energy and also supplies the latent-charge response potential.

CubeS2 spreading remains non-separable. The Form-B Green multiplier remains separable at rank at most the number of SOG terms; the present distributed loop reads the already validated global Green tables for its local mode range.

## CUDA and FFT boundary

Local particle work is CUDA-resident, one rank per GPU. `Grid3d` communication, brick-to-pencil remap, and LAMMPS `FFT3d` currently use host buffers. The FFT implementation is selected when LAMMPS is built:

- the original MPI executable uses KISS FFT;
- the performance executable uses FFTW3;
- neither executable currently uses a GPU-distributed FFT library.

Calling this a fully distributed GPU FFT would therefore be inaccurate. The implemented design matches the host-communication structure of PPPM/GPU: CUDA assignment/interpolation surrounds a distributed host FFT.

## Correctness result

For the 1536-atom water frame, CubeS2-6 with `phi_max=0.066`, two-rank distributed and replicated calculations agree to below `1e-10 eV` in total energy. With the FFTW3 performance executable, componentwise force RMS differences are `5.468e-8`, `5.445e-8`, and `5.473e-8 eV/Ang`, with maxima below `4.77e-7 eV/Ang`.

Against the ASE direct-k reference, the distributed result gives:

- energy difference: `0.005089 meV/atom`;
- force RMSE: `8.568e-4 eV/Ang`;
- force relative RMS: `1.0583e-3`;
- virial Frobenius relative error: `2.127e-4`.

Raw files are under `dp/dp_example/water-scan0/lmp/ase_lammps_benchmark/` with stems `phi0066_correct_final_n2_distributed` and `phi0066_distributed_vs_replicated`.

## Current performance limitations

The per-step mesh and FFT are partitioned, but several setup and staging costs remain:

- `ensure_fft_plan()` still allocates the legacy full-mesh host arrays and full Green tables on every rank before the distributed plan is created;
- GPU brick data use synchronous transfers through pageable host vectors;
- `Grid3d`, `Remap`, and `FFT3d` are host-side and are not overlapped with CUDA work;
- the reciprocal Green tables are globally allocated even though each rank reads only its local pencil.

The next performance changes, if justified by profiling, are pinned persistent brick buffers, transfer/communication overlap, rank-local Green storage, removal of the unused full legacy FFT plan in distributed mode, and evaluation of a CUDA-aware distributed FFT library.
