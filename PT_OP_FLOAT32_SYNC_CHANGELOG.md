# PT NUFFT Op Float32 Sync Notes

Date: 2026-04-08

## Scope

Synchronized SOG and LES PyTorch NUFFT custom ops to support native float32 and float64 dispatch without wrapper-level float32->float64 casting.

## Code Changes

- SOG C++ op: native float32/float64 dispatch, precision-aware plan cache, typed setpts/execute paths.
  - source/op/pt/nufft_sog_op.cc
- SOG CUDA kernels: templated float/double kernels and dual launchers.
  - source/op/pt/sog_nufft_kernels.cu
- LES C++ op: native float32/float64 dispatch, precision-aware plan cache, typed setpts/execute paths.
  - source/op/pt/nufft_les_op.cc
- LES CUDA kernels: templated float/double kernels and dual launchers.
  - source/op/pt/les_nufft_kernels.cu
- Added LES float32 custom-op vs python-fallback benchmark script.
  - dp_example/nacl/bench_les_op_float32_vs_fallback.py

## Build Validation

- Build command:
  - `cmake --build source/build --target deepmd_op_pt -j 8`
- Result:
  - `Built target deepmd_op_pt`

## Benchmark Commands

- `python -u dp_example/nacl/bench_sog_op_accuracy_speed.py`
- `python -u dp_example/nacl/bench_les_op_accuracy_speed.py`
- `python -u dp_example/nacl/bench_les_op_float32_vs_fallback.py`
- SOG float32 fallback comparison snippet executed in terminal (forward+loss+backward, nloc=96).

## Unified Accuracy/Performance Summary

### Float64 custom-op vs reference (pytorch_finufft)

| Model | Case | Max Abs Err | Max Rel Err | Custom (ms/call) | Ref (ms/call) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SOG | data_16_train (8f) | 6.106227e-16 | 3.054427e-15 | 4.105 | 54.538 | 13.29x |
| SOG | data_17_train (8f) | 8.049117e-16 | 3.478318e-15 | 4.114 | 48.578 | 11.81x |
| LES | data_16_train (8f) | 2.053913e-15 | 4.514820e-15 | 3.944 | 77.653 | 19.69x |
| LES | data_17_train (8f) | 4.440892e-15 | 7.992299e-15 | 3.908 | 46.120 | 11.80x |

### Float32 custom-op vs python fallback (e2e)

| Model | Workload | Accuracy Delta | Custom (ms/iter) | Fallback (ms/iter) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| SOG | forward+loss+backward, nloc=96 | energy/grad_latent max_abs_err <= 1e-7 (quick sanity check) | 31.157 | 76.533 | 2.46x |
| LES | forward corr + backward latent/sigma, nloc=256 | energy/grad_latent/grad_sigma all close (<=1e-6 scale) | 9.549 | 27.402 | 2.87x |

## Notes

- Root causes found during re-check:
  - SOG/LES float64 benchmark references were using centered mode indexing; corrected to FFT-order indexing to match modeord=1.
  - LES float32 fallback benchmark mistakenly set `coord.requires_grad=True`, which forced fallback path and invalidated speed conclusion.
  - LES CUDA kernels used centered `n = i - nk` indexing; corrected to FFT-order mapping.
  - cuFINUFFT type-2 execute argument order in LES path was reversed; corrected to `(out_nonuniform, in_modes)`.
  - LES type-2 plan sign updated to `isign=+1` (type-1 remains `-1`) to match pytorch_finufft path.
  - SOG/LES plan option initialization now safely falls back to available default-opts symbol (`*_default_opts`) and still enforces `modeord=1` when defaults are available.

- After fixes, LES float32 path shows both correctness parity and clear e2e speedup versus fallback in the tested workload.

## Follow-ups

- Profile LES float32 e2e hot spots (plan reuse, kernel occupancy, memory traffic) for the nloc=256 workload.
- Compare LES float32 performance over multiple nloc values (64/128/256/512) to identify crossover points.
