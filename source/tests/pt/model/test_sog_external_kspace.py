# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

torch.set_default_device("cpu")

from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.sog_model import (
    SOGEnergyModel,
)
from deepmd.pt.model.task.sog_energy_fitting import (
    SOGEnergyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)


dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = torch.device("cpu")


class TestSOGExternalKspace(unittest.TestCase):
    def test_fitting_serialize_deserialize_external_kspace(self) -> None:
        fitting = SOGEnergyFittingNet(
            var_name="energy",
            ntypes=2,
            dim_descrpt=8,
            dim_out_sr=1,
            dim_out_lr=1,
            mixed_types=True,
            external_kspace=True,
        ).to(device)

        data = fitting.serialize()
        self.assertTrue(data["external_kspace"])

        restored = SOGEnergyFittingNet.deserialize(data)
        self.assertTrue(restored.external_kspace)

    def test_forward_lower_exposes_latent_charge_and_skips_frame_corr_when_external(
        self,
    ) -> None:
        torch.manual_seed(2031)
        nf = 1
        nloc = 4
        nt = 2
        rcut = 4.0
        rcut_smth = 3.5
        sel = [8, 8]

        descriptor = DescrptSeA(
            rcut,
            rcut_smth,
            sel,
        ).to(device)
        fitting = SOGEnergyFittingNet(
            var_name="energy",
            ntypes=nt,
            dim_descrpt=descriptor.get_dim_out(),
            dim_out_sr=1,
            dim_out_lr=1,
            mixed_types=descriptor.mixed_types(),
            n_dl=2,
            external_kspace=True,
        ).to(device)
        model = SOGEnergyModel(
            descriptor=descriptor,
            fitting=fitting,
            type_map=["A", "B"],
        ).to(device)

        coord = torch.rand((nf, nloc, 3), dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 5.0
        atype = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64, device=device)

        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            coord,
            atype,
            model.get_rcut(),
            model.get_sel(),
            mixed_types=True,
            box=cell,
        )

        lower_ret = model.forward_common_lower(
            extended_coord=extended_coord,
            extended_atype=extended_atype,
            nlist=nlist,
            mapping=mapping,
            do_atomic_virial=False,
            comm_dict={"box": cell},
        )

        # external_kspace=True: no in-model frame correction
        expected_energy_redu = lower_ret["energy"].sum(dim=1)
        torch.testing.assert_close(
            lower_ret["energy_redu"],
            expected_energy_redu,
            rtol=1e-8,
            atol=1e-8,
        )

        fw_lower = model.forward_lower(
            extended_coord=extended_coord,
            extended_atype=extended_atype,
            nlist=nlist,
            mapping=mapping,
            do_atomic_virial=False,
            comm_dict={"box": cell},
        )

        self.assertIn("latent_charge", fw_lower)
        self.assertEqual(fw_lower["latent_charge"].shape[0], nf)


if __name__ == "__main__":
    unittest.main()
