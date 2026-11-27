# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the Kokkos pair style with a two-block DPA4 long-range canonical model.

``pair_style deepmd/kk`` evaluates the archive through its compact canonical
lower (``lower_input_kind == "dpa_canonical"``, ``canonical_evaluator == "graph"``),
passing the full source CSR, the positions, the cell, the MPI communicator, and
the ``DP_HaloExchange`` border plan in the main ABI.

Run ``source/install/test_lammps_kokkos.sh`` after
``source/install/test_cc_local.sh`` on a CUDA host: that job generates the
fixture with ``source/tests/infer/gen_dpa4_2l_canonical.py`` and sets
``DPA4_2L_KOKKOS_REQUIRED=1`` so that no missing prerequisite skips the test
silently.  The model is untrained: only the agreement between the one-rank and
two-rank evaluations of the same archive is asserted.
"""

from __future__ import (
    annotations,
)

import importlib.util
import json
import os
import shutil
import zipfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
from expected_ref import (
    read_expected_ref,
)
from lammps_test_utils import (
    run_mpi_pair_runner,
)
from write_lmp_data import (
    write_lmp_data,
)

_INFER = Path(__file__).resolve().parents[2] / "tests" / "infer"
_RUNNER = Path(__file__).with_name("run_mpi_pair_deepmd_dpa3_pt2.py")


@pytest.fixture(scope="module")
def kokkos_job() -> None:
    if os.environ.get("DPA4_2L_KOKKOS_REQUIRED") != "1":
        pytest.skip("Enable the dedicated CUDA Kokkos plugin regression explicitly")
    for module in ("lammps", "mpi4py"):
        if importlib.util.find_spec(module) is None:
            pytest.fail(f"Required Python package missing: {module}")
    if shutil.which("mpirun") is None:
        pytest.fail("mpirun required for the Kokkos regression")


def test_two_block_canonical_agrees_across_ranks(
    kokkos_job: None, tmp_path: Path
) -> None:
    """The two-block DPA4 long-range canonical archive yields matching E/F/V across ranks."""
    model = _INFER / "deeppot_dpa4_2l_canonical.pt2"
    assert model.is_file(), f"Generate the fixture first: {model}"
    with zipfile.ZipFile(model) as archive:
        metadata = json.loads(archive.read("model/extra/metadata.json"))
    assert metadata["lower_input_kind"] == "dpa_canonical"
    assert metadata["canonical_evaluator"] == "graph"
    assert metadata["has_message_passing"] is True
    assert metadata["has_comm_artifact"] is False
    assert metadata["canonical_requires_source_csr"] is True
    system = read_expected_ref(model.with_suffix(".expected"))["system"]
    cell = system["box"].reshape(3, 3)
    box = np.array([0, cell[0, 0], 0, cell[1, 1], 0, cell[2, 2], 0, 0, 0], dtype=float)
    data = tmp_path / "lattice.lmp"
    write_lmp_data(
        box,
        system["coord"].reshape(-1, 3),
        system["atype"].astype(int) + 1,
        data,
    )
    single, parallel = (
        run_mpi_pair_runner(
            _RUNNER,
            data,
            model,
            nprocs=ranks,
            processors="1 1 1" if ranks == 1 else f"{ranks} 1 1",
            runner_args=["--kokkos"],
            timeout=600,
        )
        for ranks in (1, 2)
    )
    assert parallel["pe"] == pytest.approx(single["pe"], rel=1e-5, abs=1e-5)
    np.testing.assert_allclose(
        parallel["forces"], single["forces"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        parallel["virials"], single["virials"], rtol=1e-4, atol=1e-5
    )
