# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

import sog as sog_lib

from deepmd.pt.model.atomic_model import (
    SOGEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from .dp_model import (
    DPModelCommon,
)
from .make_hessian_model import (
    make_hessian_model,
)
from .make_model import (
    make_model,
)

E2_PER_ANGSTROM_TO_EV = 14.3996454784255

SOGEnergyModel_ = make_model(SOGEnergyAtomicModel)


@BaseModel.register("sog_ener")
class SOGEnergyModel(DPModelCommon, SOGEnergyModel_):
    model_type = "sog_ener"
    fitting_net_type = "sog_energy"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        SOGEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False

    def enable_hessian(self) -> None:
        self.__class__ = make_hessian_model(type(self))
        self.hess_fitting_def = super(type(self), self).atomic_output_def()
        self.requires_hessian("energy")
        self._hessian_enabled = True

    @torch.jit.export
    def get_observed_type_list(self) -> list[str]:
        """Get observed types (elements) of the model during data statistics.

        Returns
        -------
        observed_type_list: a list of the observed types in this model.
        """
        type_map = self.get_type_map()
        out_bias = self.atomic_model.get_out_bias()[0]

        assert out_bias is not None, "No out_bias found in the model."
        assert out_bias.dim() == 2, "The supported out_bias should be a 2D tensor."
        assert out_bias.size(0) == len(type_map), (
            "The out_bias shape does not match the type_map length."
        )
        bias_mask = (
            torch.gt(torch.abs(out_bias), 1e-6).any(dim=-1).detach().cpu()
        )  # 1e-6 for stability

        observed_type_list: list[str] = []
        for i in range(len(type_map)):
            if bias_mask[i]:
                observed_type_list.append(type_map[i])
        return observed_type_list

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if self._hessian_enabled:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def

    def _build_sog_lib_direct_kernel(
        self,
        fitting: Any,
        runtime_device: torch.device,
        real_dtype: torch.dtype,
    ) -> Any:
        bw_init = torch.sqrt(
            torch.clamp(
                fitting.bandwidth.detach().to(device=runtime_device, dtype=real_dtype),
                min=torch.finfo(real_dtype).tiny,
            )
        )

        kernel = sog_lib.Sog(
            sog_arguments={
                "use_atomwise": False,
                "n_dl": float(fitting.n_dl),
                "amp": fitting.amp.detach().to(device=runtime_device, dtype=real_dtype),
                "bandwidth": bw_init,
                "remove_self_interaction": bool(fitting.remove_self_interaction),
                "nufft": False,
                "use_nufft": False,
                "norm_factor": E2_PER_ANGSTROM_TO_EV,
                "trainable_kernel": False,
            },
            r_cut=float(self.get_rcut()),
        )

        kernel.gaussian.amp = fitting.amp
        kernel.gaussian.bandwidth = fitting.bandwidth
        return kernel

    def _compute_sog_frame_correction_bundle(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        *,
        need_force: bool,
        need_virial: bool,
    ) -> dict[str, torch.Tensor]:
        fitting = self.get_fitting_net()
        runtime_device = coord.device
        real_dtype = coord.dtype

        coord = coord.to(device=runtime_device, dtype=real_dtype)
        latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)

        nf, nloc, _ = coord.shape
        nq = latent_charge.shape[-1]
        batch = torch.arange(nf, device=runtime_device, dtype=torch.int64).repeat_interleave(nloc)

        kernel = self._build_sog_lib_direct_kernel(
            fitting,
            runtime_device,
            real_dtype,
        )

        def _corr_redu(positions: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
            corr = kernel(
                positions=positions.reshape(nf * nloc, 3),
                cell=box,
                batch=batch,
                latent_charges=charges.reshape(nf * nloc, nq),
                compute_energy=True,
                compute_bec=False,
            )["E_lr"]
            assert corr is not None
            return corr.reshape(nf, 1)

        if need_force or need_virial:
            coord_for_grad = (
                coord
                if coord.requires_grad
                else coord.detach().clone().requires_grad_(True)
            )
            corr_redu = _corr_redu(coord_for_grad, latent_charge)

            grad_result = torch.autograd.grad(
                [corr_redu],
                [coord_for_grad],
                grad_outputs=[torch.ones_like(corr_redu)],
                create_graph=self.training,
                retain_graph=True,
                allow_unused=True,
            )[0]
            force_local = (
                -grad_result
                if grad_result is not None
                else torch.zeros_like(coord_for_grad)
            )
            out: dict[str, torch.Tensor] = {"corr_redu": corr_redu}
            out["force_local"] = force_local
            if need_virial:
                out["virial_local"] = torch.einsum("bai,baj->baij", force_local, coord).reshape(
                    nf, nloc, 1, 9
                )
            return out

        out = {"corr_redu": _corr_redu(coord, latent_charge)}
        return out

    def _apply_frame_correction_lower(
        self,
        model_ret: dict[str, torch.Tensor],
        extended_coord: torch.Tensor,
        nlist: torch.Tensor,
        box: torch.Tensor | None,
        do_atomic_virial: bool,
    ) -> dict[str, torch.Tensor]:
        if box is None or "latent_charge" not in model_ret:
            return model_ret

        nf, nloc, _ = nlist.shape
        nall = extended_coord.shape[1]
        coord_local = extended_coord[:, :nloc, :]
        box_local = box.view(nf, 3, 3)
        latent_charge = model_ret["latent_charge"]
        need_force = self.do_grad_r("energy") or self.do_grad_c("energy")
        need_virial = self.do_grad_c("energy")

        latent_charge_runtime = latent_charge[:, :nloc, :]
        corr_bundle = self._compute_sog_frame_correction_bundle(
            coord_local,
            latent_charge_runtime,
            box_local,
            need_force=need_force,
            need_virial=need_virial,
        )
        corr_redu = corr_bundle["corr_redu"]
        model_ret["energy_redu"] = model_ret["energy_redu"] + corr_redu.to(
            model_ret["energy_redu"].dtype
        ).view_as(model_ret["energy_redu"])

        corr_virial_local: torch.Tensor | None = None
        if need_force:
            corr_force_local = corr_bundle["force_local"].to(coord_local.dtype)
            if need_virial:
                corr_virial_local = corr_bundle["virial_local"].to(
                    corr_force_local.dtype
                )

        if need_force:
            corr_force_ext = torch.zeros(
                (nf, nall, 3),
                dtype=corr_force_local.dtype,
                device=corr_force_local.device,
            )
            corr_force_ext[:, :nloc, :] = corr_force_local
            if "energy_derv_r" in model_ret:
                model_ret["energy_derv_r"] = model_ret[
                    "energy_derv_r"
                ] + corr_force_ext.unsqueeze(-2).to(model_ret["energy_derv_r"].dtype).view_as(
                    model_ret["energy_derv_r"]
                )

            if need_virial:
                assert corr_virial_local is not None
                corr_virial_redu = corr_virial_local.sum(dim=1)
                if "energy_derv_c_redu" in model_ret:
                    model_ret["energy_derv_c_redu"] = model_ret[
                        "energy_derv_c_redu"
                    ] + corr_virial_redu.to(model_ret["energy_derv_c_redu"].dtype).view_as(
                        model_ret["energy_derv_c_redu"]
                    )
                if do_atomic_virial and "energy_derv_c" in model_ret:
                    corr_atom_virial = torch.zeros(
                        (nf, nall, 1, 9),
                        dtype=corr_virial_local.dtype,
                        device=corr_virial_local.device,
                    )
                    corr_atom_virial[:, :nloc, :, :] = corr_virial_local
                    model_ret["energy_derv_c"] = model_ret[
                        "energy_derv_c"
                    ] + corr_atom_virial.to(model_ret["energy_derv_c"].dtype).view_as(
                        model_ret["energy_derv_c"]
                    )

        return model_ret

    @torch.jit.export
    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extra_nlist_sort: bool = False,
        extended_coord_corr: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.do_grad_r("energy") or self.do_grad_c("energy"):
            extended_coord = extended_coord.requires_grad_(True)

        nframes, _ = extended_atype.shape[:2]
        extended_coord = extended_coord.view(nframes, -1, 3)
        nlist = self.format_nlist(
            extended_coord,
            extended_atype,
            nlist,
            extra_nlist_sort=extra_nlist_sort,
        )
        cc_ext, _, fp, ap, input_prec = self._input_type_cast(
            extended_coord,
            fparam=fparam,
            aparam=aparam,
        )

        atomic_ret = self.atomic_model.forward_common_atomic(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fp,
            aparam=ap,
            comm_dict=comm_dict,
        )

        runtime_box = box
        if runtime_box is None and comm_dict is not None and "box" in comm_dict:
            runtime_box = comm_dict["box"]

        model_ret = fit_output_to_model_output(
            atomic_ret,
            self.atomic_output_def(),
            cc_ext,
            do_atomic_virial=do_atomic_virial,
            create_graph=self.training,
            mask=atomic_ret["mask"] if "mask" in atomic_ret else None,
            extended_coord_corr=extended_coord_corr,
        )
        model_ret = self._output_type_cast(model_ret, input_prec)
        return self._apply_frame_correction_lower(
            model_ret,
            cc_ext,
            nlist,
            runtime_box,
            do_atomic_virial,
        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        cc, bb, fp, ap, input_prec = self._input_type_cast(
            coord, box=box, fparam=fparam, aparam=aparam
        )
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            cc,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=True,
            box=bb,
        )
        comm_dict: dict[str, torch.Tensor] | None = None
        if bb is not None:
            comm_dict = {"box": bb}
        model_predict_lower = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fp,
            aparam=ap,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            box=bb,
        )
        model_ret = communicate_extended_output(
            model_predict_lower,
            self.model_output_def(),
            mapping,
            do_atomic_virial=do_atomic_virial,
        )
        model_ret = self._output_type_cast(model_ret, input_prec)
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -2
                    )
            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            if self._hessian_enabled:
                model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        box: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            box=box,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-2)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict
