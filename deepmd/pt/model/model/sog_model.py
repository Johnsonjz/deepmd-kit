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
        runtime_device: torch.device,
        real_dtype: torch.dtype,
    ) -> Any:
        fitting = self.atomic_model.fitting_net
        bw2_runtime = fitting.bandwidth.to(device=runtime_device, dtype=real_dtype)
        amp_internal_runtime = fitting.amp.to(
            device=runtime_device,
            dtype=real_dtype,
        )

        nlayers = getattr(self.atomic_model.descriptor, "nlayers", 1) if hasattr(self.atomic_model, "descriptor") else 1
        # Auto-detect FFT vs direct:
        # Explicit n_dl without cubes2_phi_max → direct k-sum (training/validation).
        _has_n_dl = getattr(fitting, "n_dl", None) is not None
        _has_phi = getattr(fitting, "cubes2_phi_max", None) is not None
        _use_fft = bool(getattr(fitting, "use_cubes2_fft", False))
        if _has_n_dl and not _has_phi:
            _use_fft = False
        # Build sog_arguments dict
        sog_args: dict = {
            "use_atomwise": False,
            "amp": amp_internal_runtime,
                "bandwidth": bw2_runtime,
                "kernel_param_mode": "internal",
                "kernel_tensor_mode": "external",
                "remove_self_interaction": bool(fitting.remove_self_interaction),
                "nufft": False,
                "use_nufft": False,
                "use_cubes2_fft": _use_fft,
                "nlayers": nlayers,
                "norm_factor": E2_PER_ANGSTROM_TO_EV,
                "trainable_kernel": False,
                "b": float(fitting.b),
            }
        # Prefer cubes2_phi_max (new API), fall back to n_dl (legacy)
        if getattr(fitting, "cubes2_phi_max", None) is not None:
            sog_args["cubes2_phi_max"] = float(fitting.cubes2_phi_max)
        elif getattr(fitting, "n_dl", None) is not None:
            sog_args["n_dl"] = float(fitting.n_dl)
        # else: auto-default from SOG lib's Table III
        # Optional charge neutrality penalty (None = disabled, use physical k=0 instead)
        if getattr(fitting, "charge_neutral_lambda", None) is not None:
            sog_args["charge_neutral_lambda"] = float(fitting.charge_neutral_lambda)

        kernel = sog_lib.Sog(
            sog_arguments=sog_args,
            r_cut=float(self.get_rcut()),
        )
        return kernel

    def _compute_sog_frame_correction_bundle(
        self,
        base_coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        nloc: int,
        *,
        need_force: bool,
        need_virial: bool,
    ) -> dict[str, torch.Tensor]:
        # base_coord: [nf, nall, 3] — the graph tensor that BOTH the local positions
        #   (a slice) AND latent_charge (via the fitting net) derive from. Differentiating
        #   E_lr w.r.t. base_coord therefore captures the explicit ∂E_lr/∂r AND the
        #   charge-response ∂E_lr/∂q·∂q/∂r term (conservative force). Grad'ing w.r.t. a
        #   detached slice (the old behaviour) dropped the charge-response term.
        runtime_device = base_coord.device
        real_dtype = base_coord.dtype

        latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)
        if need_virial:
            # Make box a leaf so a single backward yields ∂E_lr/∂h (the box-strain
            # virial Ξ_rec) alongside the force — avoids a second kernel call.
            box = box.detach().clone().requires_grad_(True)

        nf, nall, _ = base_coord.shape
        nq = latent_charge.shape[-1]
        batch = torch.arange(nf, device=runtime_device, dtype=torch.int64).repeat_interleave(nloc)

        kernel = self._build_sog_lib_direct_kernel(
            runtime_device,
            real_dtype,
        )

        # E_lr is the reciprocal-space sum over the LOCAL atoms; positions and charges
        # are slices/functions of base_coord so autograd flows back through base_coord.
        positions = base_coord[:, :nloc, :]
        q_loc = latent_charge[:, :nloc, :]
        corr = kernel(
            positions=positions.reshape(nf * nloc, 3),
            cell=box,
            batch=batch,
            latent_charges=q_loc.reshape(nf * nloc, nq),
            compute_energy=True,
            compute_bec=False,
        )["E_lr"]
        assert corr is not None
        corr_redu = corr.reshape(nf, 1)
        out: dict[str, torch.Tensor] = {"corr_redu": corr_redu}

        if need_force or need_virial:
            grad_inputs = [base_coord] + ([box] if need_virial else [])
            grads = torch.autograd.grad(
                [corr_redu],
                grad_inputs,
                grad_outputs=[torch.ones_like(corr_redu)],
                create_graph=self.training,
                retain_graph=True,
            )
            force_ext = -grads[0]
            assert force_ext is not None
            out["force_ext"] = force_ext  # [nf, nall, 3] — full charge-response force
            if need_virial:
                box_grad = grads[1]
                assert box_grad is not None
                out["box_grad"] = box_grad  # ∂E_lr/∂h at fixed r & q (→ Ξ_rec)
                out["box_leaf"] = box       # the box tensor h that box_grad is w.r.t.
        return out

    def _compute_sog_strain_virial(
        self,
        base_coord: torch.Tensor,
        force_ext: torch.Tensor,
        box_grad: torch.Tensor,
        box_leaf: torch.Tensor,
    ) -> torch.Tensor:
        # Complete LR virial (DPLR Eq. 22-24) via decomposition, NO extra kernel call:
        #   position part  F_full ⊗ r   — F_full is the conservative LR force, so this
        #                                  carries the charge-response term Ξ_c; and
        #   box part       -∂E_lr/∂h · h  (Ξ_rec), from the same backward as the force.
        # Signs match deepmd's energy_derv_c convention; verified vs strain finite diff.
        nf = base_coord.shape[0]
        pos_vir = torch.einsum("fak,faj->fkj", force_ext, base_coord)  # [nf,3,3]
        box_vir = -torch.einsum("fga,fgb->fab", box_grad, box_leaf)     # [nf,3,3]
        return (pos_vir + box_vir).reshape(nf, 1, 9)



    def _apply_frame_correction_lower(
        self,
        model_ret: dict[str, torch.Tensor],
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        box: torch.Tensor | None,
        do_atomic_virial: bool,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        fitting = self.atomic_model.fitting_net
        if fitting is not None and bool(fitting.external_kspace):
            return model_ret

        # TorchScript export is used for frozen inference models where the
        # long-range correction is expected to be provided externally.
        if torch.jit.is_scripting():
            return model_ret

        if box is None or "latent_charge" not in model_ret:
            return model_ret

        nf, nloc, _ = nlist.shape
        box_local = box.view(nf, 3, 3)
        latent_charge = model_ret["latent_charge"]
        need_force = self.do_grad_r("energy") or self.do_grad_c("energy")
        need_virial = self.do_grad_c("energy")

        corr_bundle = self._compute_sog_frame_correction_bundle(
            extended_coord,
            latent_charge,
            box_local,
            nloc,
            need_force=need_force,
            need_virial=need_virial,
        )
        corr_redu = corr_bundle["corr_redu"]
        model_ret["energy_redu"] = model_ret["energy_redu"] + corr_redu.to(
            model_ret["energy_redu"].dtype
        ).view_as(model_ret["energy_redu"])

        if need_force:
            # force_ext: [nf, nall, 3], full charge-response force over extended atoms.
            # It is added to the extended energy_derv_r; the framework later reduces
            # ghost contributions to local atoms via communicate_extended_output.
            force_ext = corr_bundle["force_ext"]
            if "energy_derv_r" in model_ret:
                model_ret["energy_derv_r"] = model_ret[
                    "energy_derv_r"
                ] + force_ext.unsqueeze(-2).to(model_ret["energy_derv_r"].dtype).view_as(
                    model_ret["energy_derv_r"]
                )

            if need_virial:
                # Complete LR virial (DPLR Ξ_rec + Ξ_c) = F_full⊗r + box-strain, no
                # atomic-model recompute. communicate_extended_output re-sums the atomic
                # energy_derv_c into the reduced virial, so inject the global term into
                # the atomic virial (spread over local atoms); also set the redu key for
                # the forward_lower path that skips communicate.
                virial_redu = self._compute_sog_strain_virial(
                    extended_coord,
                    corr_bundle["force_ext"],
                    corr_bundle["box_grad"],
                    corr_bundle["box_leaf"],
                )  # [nf, 1, 9]
                if "energy_derv_c" in model_ret:
                    nall = model_ret["energy_derv_c"].shape[1]
                    corr_c = torch.zeros(
                        (nf, nall, 1, 9),
                        dtype=model_ret["energy_derv_c"].dtype,
                        device=model_ret["energy_derv_c"].device,
                    )
                    corr_c[:, :nloc, :, :] = (virial_redu / nloc).unsqueeze(1).to(
                        corr_c.dtype
                    )
                    model_ret["energy_derv_c"] = model_ret["energy_derv_c"] + corr_c
                if "energy_derv_c_redu" in model_ret:
                    model_ret["energy_derv_c_redu"] = model_ret[
                        "energy_derv_c_redu"
                    ] + virial_redu.to(model_ret["energy_derv_c_redu"].dtype).view_as(
                        model_ret["energy_derv_c_redu"]
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
            extended_atype,
            nlist,
            runtime_box,
            do_atomic_virial,
            mapping=mapping,
            fparam=fp,
            aparam=ap,
            comm_dict=comm_dict,
        )

    @torch.jit.export
    def charge_response_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        grad_charge: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # DPLR-style charge-response feedback (analog of DipoleChargeModifier::compute).
        # Given the external per-atom potential v_i = ∂E_k/∂q_i (grad_charge,
        # [nf, nloc, nq]) from the SOG kspace solver, compute the terms the fixed-q
        # kspace omits:
        #   force_j = -Σ_i v_i ∂q_i/∂r_j     (charge-response force, [nf, nall, 3])
        #   virial  = force ⊗ r              (Ξ_c; the box-strain Ξ_rec is already in the
        #                                     kspace analytical virial)
        nframes = extended_atype.shape[0]
        extended_coord = extended_coord.view(nframes, -1, 3).requires_grad_(True)
        nlist = self.format_nlist(extended_coord, extended_atype, nlist)
        cc_ext, _, _, _, input_prec = self._input_type_cast(extended_coord)
        atomic_ret = self.atomic_model.forward_common_atomic(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
        )
        latent_charge = atomic_ret["latent_charge"]  # [nf, nloc, nq]
        nloc = nlist.shape[1]
        gc = grad_charge.to(cc_ext.dtype).reshape(nframes, nloc, -1)
        scalar = (gc * latent_charge[:, :nloc, :]).sum()
        grad_outputs = torch.jit.annotate(
            list[torch.Tensor | None], [torch.ones_like(scalar)]
        )
        grads = torch.autograd.grad(
            [scalar],
            [cc_ext],
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=True,
        )
        fe = grads[0]
        assert fe is not None
        force_ext = -fe
        virial = torch.einsum("fak,faj->fkj", force_ext, cc_ext).reshape(nframes, 9)
        return {
            "charge_response_force": force_ext,
            "charge_response_virial": virial,
        }

    @torch.jit.export
    def forward_lower_energy_charge(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # FUSED charge-response path (Tier-2): energy-only + charge forward. Returns the reduced
        # energy (differentiable to the returned coord leaf) and latent_charge, but does NOT compute
        # extended_force (energy_derv_r). The C++ side retains this graph and runs ONE combined
        # backward   grad( energy.sum() + Σ_i v_i q_i , coord_leaf )  ->  total (short-range +
        # charge-response) force, eliminating the model's internal force backward (the pair's
        # forward_lower would otherwise autograd.grad(energy, coord) for extended_force, and the fix
        # would autograd.grad((v·q), coord) again — two backwards for what fuses into one). Same
        # weights / same descriptor+fitting path as forward_lower — only the returned derivatives differ.
        nframes = extended_atype.shape[0]
        extended_coord = extended_coord.view(nframes, -1, 3)
        nlist = self.format_nlist(extended_coord, extended_atype, nlist)
        cc_ext, _, _, _, input_prec = self._input_type_cast(extended_coord)
        atomic_ret = self.atomic_model.forward_common_atomic(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
        )
        atom_energy = atomic_ret["energy"]           # [nf, nloc, 1], differentiable to the input coord
        energy_redu = atom_energy.sum(dim=1)         # [nf, 1]
        latent_charge = atomic_ret["latent_charge"]  # [nf, nloc, nq]
        return {
            "energy": energy_redu,
            "latent_charge": latent_charge,
        }

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
            extra_nlist_sort=False,
            extended_coord_corr=None,
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
            if "latent_charge" in model_ret:
                model_predict["latent_charge"] = model_ret["latent_charge"]
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
            extended_coord_corr=None,
            box=box,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if "latent_charge" in model_ret:
                model_predict["latent_charge"] = model_ret["latent_charge"]
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
