# SOG/LES Developer Workflow (Build, Input, Validation, Debug)

This document records a reproducible workflow for developers working on
DeepMD LAMMPS plugin styles:

- kspace_style sog
- kspace_style les

Scope:

1. Build DeepMD LAMMPS plugin
2. Build plugin-enabled LAMMPS executable
3. Prepare input scripts
4. Run accuracy and speed validation
5. Debug common failures

This guide is intentionally command-oriented so it can be copy-pasted.

## 1. Known Good Paths

Workspace root:

- /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel

DeepMD plugin build dir:

- /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable

LAMMPS source root:

- /data/zyjin/lammps/lammps_stable_latest

LAMMPS executable (plugin-enabled):

- /data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp

Plugin shared library:

- /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so

Recommended Python interpreter for validation scripts:

- /home/zyjin/anaconda3/envs/dp_devel/bin/python

Reason: in persistent shells, conda activation can be polluted by repeated PATH
rewrites, so using absolute interpreter path avoids accidental base_env runtime.

## 2. Environment Sanity Check (Do This First)

Run:

```bash
/home/zyjin/anaconda3/envs/dp_devel/bin/python - <<'PY'
import sys
import deepmd
import torch
print('python_prefix =', sys.prefix)
print('deepmd_pkg    =', deepmd.__file__)
print('deepmd_ver    =', deepmd.__version__)
print('torch_ver     =', torch.__version__)
PY
```

Expected:

- python_prefix contains /home/zyjin/anaconda3/envs/dp_devel
- deepmd_pkg points to editable source tree under deepmd-kit-devel/deepmd
  or to the intended dp_devel installation

If this check fails, do not continue with validation runs.

## 3. Build DeepMD Plugin (libdeepmd_lmp.so)

### 3.1 Configure (first time only)

```bash
cd /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel

cmake -S source -B build/sog_lmp_stable \
  -DENABLE_PYTORCH=ON \
  -DENABLE_TENSORFLOW=OFF \
  -DBUILD_CPP_IF=ON \
  -DBUILD_PY_IF=OFF \
  -DLAMMPS_SOURCE_ROOT=/data/zyjin/lammps/lammps_stable_latest \
  -DCMAKE_PREFIX_PATH=/home/zyjin/anaconda3/envs/dp_devel/lib/python3.13/site-packages/torch/share/cmake
```

### 3.2 Build plugin target

```bash
cd /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel
cmake --build build/sog_lmp_stable --target deepmd_lmp -j 12
```

### 3.3 Verify artifact

```bash
ls -l /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so
```

## 4. Build Plugin-Enabled LAMMPS

### 4.1 Configure

```bash
cd /data/zyjin/lammps/lammps_stable_latest

cmake -S cmake -B build_sog_bench_plugin \
  -DBUILD_MPI=no \
  -DBUILD_OMP=no \
  -DBUILD_SHARED_LIBS=OFF \
  -DPKG_PLUGIN=yes \
  -DPKG_KSPACE=yes \
  -DFFT=KISS \
  -DCMAKE_BUILD_TYPE=Release
```

### 4.2 Build

```bash
cd /data/zyjin/lammps/lammps_stable_latest
cmake --build build_sog_bench_plugin -j 12
```

### 4.3 Verify executable

```bash
ls -l /data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp
```

## 5. Runtime Smoke Check (Plugin Load)

Create a minimal input that loads plugin and confirms style registration.

```lammps
units           metal
plugin          load /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so
atom_style      charge
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 5.0 5.0 5.0
mass            1 1.0
set             atom 1 charge 0.1
pair_style      coul/long 6.0
pair_coeff      * *
kspace_style    sog 1.0e-6 amp 0.1 bandwidth 0.5
run             0
```

Run:

```bash
/data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp -in in.smoke.lmp
```

## 6. Input Script Templates

### 6.1 SOG with explicit n_dl

```lammps
plugin load /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so

read_data stable.data
pair_style coul/long 0.5
pair_coeff * *

kspace_style sog 1.0e-6 \
  n_dl 3.0 \
  b 1.6297670882677646 sigma 3.633717409009413 m 12 \
  remove_self_interaction yes \
  amp 4.047320470826997 4.047320470826997 4.047320470826997 4.047320470826997 \
      4.047320470826997 4.047320470826997 4.047320470826997 4.047320470826997 \
      4.047320470826997 4.047320470826997 4.047320470826997 4.047320470826997 \
  bandwidth 2.176273189386266 5.670674491580614 9.729256862016847 15.003529586954505 \
            22.673081210874987 36.29537380969535 59.15300569136901 96.40562184791224 \
            157.1187096117105 256.06690187627055 417.329409072649 680.1497358728163
```

### 6.2 SOG with model-driven n_dl

This mode uses n_dl parsed from pair_style deepmd model metadata.

```lammps
plugin load /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so

pair_style deepmd /path/to/model.json latent_charge_to_q yes
pair_coeff * * Mg O Al Au

kspace_style sog 1.0e-6 \
  n_dl_from_model yes \
  b 1.6297670882677646 sigma 3.633717409009413 m 12 \
  remove_self_interaction yes \
  amp ... \
  bandwidth ...
```

Priority semantics:

- explicit n_dl in kspace_style has highest priority
- if explicit n_dl and n_dl_from_model yes are both set, explicit n_dl is used
- n_dl_from_model yes is strict and errors out when model n_dl is missing
- when explicit n_dl is omitted and n_dl_from_model is not strict, code attempts
  import then falls back to default

### 6.3 LES template

```lammps
plugin load /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so

read_data stable.data
pair_style coul/long 0.5
pair_coeff * *

kspace_style les 1.0e-6 n_dl 3.0 sigma 1.692414291816906 remove_self_interaction yes
```

## 7. Accuracy Validation Workflow (Au-MgO)

Use explicit interpreter path to avoid shell contamination.

### 7.1 SOG Stage1 accuracy

```bash
/home/zyjin/anaconda3/envs/dp_devel/bin/python \
  /data/zyjin/dp_pt/dp_example/au-mgo/lmp/run_au_mgo_sog_validation.py \
  --work-dir /data/zyjin/dp_pt/dp_example/au-mgo/lmp/accuracy_check_YYYYMMDD \
  --lmp-exe /data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp \
  --plugin-so /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so \
  --skip-stage2 \
  --stage1-modes neutralized_q,raw_q
```

### 7.2 LES Stage1 accuracy

```bash
/home/zyjin/anaconda3/envs/dp_devel/bin/python \
  /data/zyjin/dp_pt/dp_example/au-mgo/lmp/run_au_mgo_les_validation.py \
  --work-dir /data/zyjin/dp_pt/dp_example/au-mgo/lmp/accuracy_check_les_YYYYMMDD \
  --lmp-exe /data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp \
  --plugin-so /data/zyjin/dp_pt/dp_devel/deepmd-kit-devel/build/sog_lmp_stable/lmp/plugin/libdeepmd_lmp.so \
  --skip-stage2 \
  --stage1-modes neutralized_q,raw_q
```

### 7.3 Output files

- SOG report: work-dir/validation_report.json
- LES report: work-dir/validation_report_les.json

## 8. Speed Benchmark Workflow (5000 steps)

A practical approach is to reuse the existing LES/SOG input pair.

Reference inputs:

- /data/zyjin/dp_pt/dp_example/au-mgo/lmp/compare_les_sog_md_20260522/in.les.lmp
- /data/zyjin/dp_pt/dp_example/au-mgo/lmp/compare_les_sog_md_20260522/in.sog.lmp

Run:

```bash
DST=/data/zyjin/dp_pt/dp_example/au-mgo/lmp/compare_les_sog_md_YYYYMMDD
mkdir -p "$DST"
cp /data/zyjin/dp_pt/dp_example/au-mgo/lmp/compare_les_sog_md_20260522/in.les.lmp "$DST"/
cp /data/zyjin/dp_pt/dp_example/au-mgo/lmp/compare_les_sog_md_20260522/in.sog.lmp "$DST"/

cd "$DST"
/data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp -in in.les.lmp -log log.les.lammps > out.les.txt 2>&1
/data/zyjin/lammps/lammps_stable_latest/build_sog_bench_plugin/lmp -in in.sog.lmp -log log.sog.lammps > out.sog.txt 2>&1
```

Parse loop time from logs and report:

- LES loop time
- SOG loop time
- SOG/LES ratio
- Kspace time and percentage

## 9. Debug Playbook

### 9.1 Wrong Python environment (most common)

Symptom:

- Prompt says dp_devel, but accuracy is wildly off
- deepmd version/path unexpectedly points to base_env

Check:

```bash
which python
python - <<'PY'
import sys, deepmd
print(sys.prefix)
print(deepmd.__file__)
print(deepmd.__version__)
PY
```

Fix:

- Use absolute interpreter for all validation commands:
  /home/zyjin/anaconda3/envs/dp_devel/bin/python

### 9.2 Plugin not loaded or style not found

Symptom:

- Unrecognized kspace style sog/les

Check:

- plugin load line exists in input
- libdeepmd_lmp.so path is correct
- lmp binary was built with PKG_PLUGIN and PKG_KSPACE enabled

### 9.3 ABI mismatch between plugin and LAMMPS

Symptom:

- plugin loads fail with symbol or C++ runtime errors

Fix:

- rebuild plugin against the same LAMMPS source tree used for executable
- avoid mixing plugin from another build tree

### 9.4 latent_charge missing in model outputs

Current validation scripts include a compatibility fallback:

- try top-level model output latent_charge
- if absent, query forward_common_lower path

Files:

- /data/zyjin/dp_pt/dp_example/au-mgo/lmp/run_au_mgo_sog_validation.py
- /data/zyjin/dp_pt/dp_example/au-mgo/lmp/run_au_mgo_les_validation.py

### 9.5 n_dl_from_model behavior checks

- strict mode: n_dl_from_model yes must find model n_dl or it errors
- explicit n_dl still overrides model import

Use this to verify expected control flow during debugging.

## 10. Developer Checklist (Short Form)

1. Run environment sanity check with absolute dp_devel python.
2. Build deepmd_lmp in build/sog_lmp_stable.
3. Build LAMMPS in build_sog_bench_plugin with plugin and kspace packages.
4. Run smoke input and confirm plugin styles are registered.
5. Run Stage1 accuracy for SOG and LES with absolute dp_devel python.
6. Run 5000-step speed benchmark with the same lmp binary and plugin.
7. Archive report files and loop-time summaries with date-stamped work dirs.
