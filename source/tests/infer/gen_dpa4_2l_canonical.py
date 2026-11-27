# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the two-block DPA4 long-range canonical fixture for the LAMMPS tests.

The archive carries the SOG long-range module and a two-block DPA4 descriptor
deployed on the unified ``dpa_canonical`` lower (``canonical_evaluator="graph"``).
Its main ABI takes the node positions, the cells, the MPI communicator, and the
``BorderPlan`` halo exchange (``has_comm_artifact=False``,
``has_message_passing=True``, ``canonical_requires_source_csr=True``).

The model is untrained: the LAMMPS regression compares the one-rank and
two-rank evaluations of the same archive across ``pair_style deepmd/kk``.
Run from the repository root with the CUDA build installed::

    python source/tests/infer/gen_dpa4_2l_canonical.py
"""

from __future__ import (
    annotations,
)

import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
    write_expected_ref,
)

CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 8,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float32",
        "seed": 7,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float32",
        "seed": 7,
    },
    "long_range": {
        "type": "latent_charge",
        "channels": 1,
        "charge_net": {
            "neuron": [32, 32],
            "activation_function": "silu",
            "precision": "float32",
            "seed": 7,
        },
        "kernel": {
            "type": "sog",
            "n_gaussian": 7,
            "trainable": True,
        },
        "periodic": True,
        "kspace_tolerance": 1e-8,
        "k_grid_mode": "ellipsoid",
        "k_grid_bounds": [4, 4, 4],
    },
}

_SPACING = 3.0
_CELLS = 3
_NATOMS = _CELLS**3
_GRID = np.stack(
    np.meshgrid(*(np.arange(_CELLS),) * 3, indexing="ij"), axis=-1
).reshape(_NATOMS, 3)
_RNG = np.random.default_rng(20261123)
_COORDS = ((_GRID + 0.5) * _SPACING + _RNG.uniform(-0.2, 0.2, (_NATOMS, 3))).reshape(
    -1,
)
_CELL = (np.eye(3) * (_SPACING * _CELLS)).reshape(-1)
_ATYPES = (np.arange(_NATOMS) % 2).astype(np.int32)


def _metadata(path: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("model/extra/metadata.json"))


def main() -> None:
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.utils.canonical_graph import (
        CANONICAL_LOWER_KIND,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    ensure_inductor_compiler()
    load_custom_ops()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.join(base_dir, "deeppot_dpa4_2l_canonical")
    model = get_model(CONFIG).eval()
    model.atomic_model.enable_compression(0.5)
    print(f"Exporting to {stem}.pt2 (lower_kind='auto') ...")  # noqa: T201
    deserialize_to_file(
        f"{stem}.pt2",
        {"model": model.serialize()},
        lower_kind="auto",
        do_atomic_virial=True,
    )
    metadata = _metadata(f"{stem}.pt2")
    assert metadata["lower_input_kind"] == CANONICAL_LOWER_KIND, metadata
    assert metadata["canonical_evaluator"] == "graph", metadata
    assert metadata["has_message_passing"] is True, metadata
    assert metadata["has_comm_artifact"] is False, metadata
    assert metadata["canonical_requires_source_csr"] is True, metadata
    print("Export done.")  # noqa: T201
    write_expected_ref(
        f"{stem}.expected",
        sections={
            "system": {
                "coord": _COORDS,
                "box": _CELL,
                "atype": _ATYPES,
            }
        },
        source_script="source/tests/infer/gen_dpa4_2l_canonical.py",
    )
    print(f"Wrote {stem}.expected")  # noqa: T201


if __name__ == "__main__":
    main()
