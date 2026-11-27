# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact canonical deployment and BorderPlan exchange of DPA4 long-range models.

Under the unified ``dpa_canonical`` lower, DPA4 energy models (1-block and
multi-block, with or without an operator-eligible ``LatentChargeLongRange``
module) evaluate on ``canonical_evaluator == "graph"`` with their halo exchange
(``deepmd::border_exchange``) and long-range operator (``deepmd::latent_charge_sum``)
in the main ABI (``has_comm_artifact == False``).
"""

from __future__ import (
    annotations,
)

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.pt_expt.kernels.long_range import (
    op_available as long_range_op_available,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    canonical_lower_evaluator,
)
from deepmd.pt_expt.utils.canonical_graph import (
    CANONICAL_LOWER_KIND,
    canonical_graph_from_neighbor_graph,
)
from deepmd.pt_expt.utils.serialization import (
    _needs_with_comm_artifact,
    _trace_and_export,
)


def _dpa4_config(
    *,
    n_blocks: int = 2,
    with_long_range: bool = True,
    precision: str = "float32",
) -> dict:
    cfg: dict = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "sel": 20,
            "rcut": 4.0,
            "channels": 16,
            "n_radial": 8,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": n_blocks,
            "precision": precision,
            "seed": 1,
        },
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [16],
            "precision": precision,
            "seed": 1,
        },
    }
    if with_long_range:
        cfg["long_range"] = {
            "type": "latent_charge",
            "channels": 1,
            "charge_net": {
                "neuron": [16],
                "activation_function": "silu",
                "precision": precision,
                "seed": 1,
            },
            "kernel": {
                "type": "sog",
                "n_gaussian": 4,
                "trainable": True,
            },
            "periodic": True,
            "kspace_tolerance": 1e-8,
            "k_grid_mode": "pme",
            "pme_spacing": 0.6,
            "pme_order": 6,
            "mesh_backend": "midtown",
        }
    return cfg


def test_dpa4_long_range_supports_edge_parallel() -> None:
    """DPA4 with LatentChargeLongRange delegates multi-rank support and compression no-op cleanly."""
    model = get_model(
        _dpa4_config(n_blocks=2, with_long_range=True, precision="float64")
    )
    assert model.atomic_model.has_message_passing_across_ranks() is True
    assert model.atomic_model.supports_edge_parallel() is True
    assert model.compression_needs_min_nbor_dist() is False
    model.atomic_model.enable_compression(0.5)
    assert _needs_with_comm_artifact(model, CANONICAL_LOWER_KIND) is False
    assert _needs_with_comm_artifact(model, "graph") is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or not long_range_op_available(),
    reason="compact canonical long-range export requires CUDA custom ops",
)
@pytest.mark.parametrize("n_blocks", [1, 2])
def test_dpa4_long_range_selects_and_traces_canonical_lower(n_blocks: int) -> None:
    """Both 1-block and 2-block DPA4 + SOG models trace on dpa_canonical with border_exchange and latent_charge_sum."""
    model = (
        get_model(_dpa4_config(n_blocks=n_blocks, with_long_range=True))
        .to("cuda")
        .eval()
    )
    assert canonical_lower_evaluator(model) == "graph"
    exported, metadata, _model_json, _keys = _trace_and_export(
        {"model": model.serialize()},
        lower_kind=CANONICAL_LOWER_KIND,
        do_atomic_virial=True,
    )
    assert metadata["lower_input_kind"] == CANONICAL_LOWER_KIND
    assert metadata["canonical_evaluator"] == "graph"
    assert metadata["has_comm_artifact"] is False
    assert metadata["has_message_passing"] is True
    assert metadata["canonical_requires_source_csr"] is True
    code = str(exported.module().code)
    assert "deepmd.border_exchange" in code
    assert "deepmd.latent_charge_sum" in code


@pytest.mark.skipif(
    not torch.cuda.is_available() or not long_range_op_available(),
    reason="compact canonical long-range evaluation requires CUDA custom ops",
)
def test_dpa4_long_range_canonical_matches_model_forward() -> None:
    """Eager forward_lower_canonical_graph with B-spline PME matches full model forward."""
    cfg = _dpa4_config(n_blocks=2, with_long_range=True, precision="float32")
    cfg["long_range"]["mesh_backend"] = "bspline"
    cfg["long_range"]["pme_order"] = 8
    cfg["long_range"]["pme_spacing"] = 0.4
    model = get_model(cfg).to("cuda").eval()

    rng = torch.Generator(device="cpu").manual_seed(11)
    coord = (torch.rand(1, 6, 3, generator=rng, dtype=torch.float64) * 5.0 + 0.5).cuda()
    atype = torch.tensor([[0, 1, 1, 0, 1, 1]], dtype=torch.int64, device="cuda")
    box_cpu = torch.tensor(
        [[6.5, 0.0, 0.0, 0.0, 6.5, 0.0, 0.0, 0.0, 6.5]], dtype=torch.float64
    )
    box_cuda = box_cpu.cuda()

    ref = model(coord, atype, box=box_cuda, do_atomic_virial=True)
    ng = build_neighbor_graph(
        coord.to(torch.float32),
        atype,
        box_cuda.to(torch.float32),
        model.get_rcut(),
        canonicalize=True,
    )
    ng = dataclasses.replace(ng, n_local=ng.n_node)
    compact = canonical_graph_from_neighbor_graph(ng)
    got = model.forward_lower_canonical_graph(
        atype.reshape(-1),
        compact.n_node,
        compact.n_local,
        compact.source,
        compact.edge_vec,
        compact.destination_row_ptr,
        compact.source_row_ptr,
        compact.source_order,
        do_atomic_virial=True,
        coord=coord.reshape(-1, 3),
        communicator=torch.zeros(1, dtype=torch.int64, device="cpu"),
        box=box_cpu,
    )
    torch.testing.assert_close(got["energy"], ref["energy"], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        got["force"].reshape_as(ref["force"]).to(torch.float64),
        ref["force"],
        rtol=5e-4,
        atol=5e-4,
    )
    torch.testing.assert_close(
        got["virial"].to(torch.float64),
        ref["virial"],
        rtol=5e-4,
        atol=5e-4,
    )
