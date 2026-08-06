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
    atomic_virial_corr,
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils import (
    env,
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
        # Persist the direct-kernel SOG object across forward calls so the
        # Gaussian's _kgrid_base_cache (integer meshgrid + zero_mask) survives.
        # Keyed by (device_type, device_index, dtype_str) — rebuild on device/dtype change.
        self._sog_direct_kernel_cache: dict = {}

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
        # ── kernel persistence: reuse across forward calls so the Gaussian's
        #     _kgrid_base_cache (integer meshgrid + zero_mask) survives ──
        cache_key = (runtime_device.type, runtime_device.index or 0, str(real_dtype))
        cached = self._sog_direct_kernel_cache.get(cache_key)
        if cached is not None:
            # Bind external amp/bandwidth to the current device+dtype (they follow the
            # model's device, which can change across save/load or DDP).
            fitting = self.atomic_model.fitting_net
            cached.gaussian.amp = fitting.amp.to(device=runtime_device, dtype=real_dtype)
            cached.gaussian.bandwidth = fitting.bandwidth.to(device=runtime_device, dtype=real_dtype)
            return cached

        fitting = self.atomic_model.fitting_net
        bw2_runtime = fitting.bandwidth.to(device=runtime_device, dtype=real_dtype)
        amp_internal_runtime = fitting.amp.to(
            device=runtime_device,
            dtype=real_dtype,
        )

        nlayers = getattr(self.atomic_model.descriptor, "nlayers", 1) if hasattr(self.atomic_model, "descriptor") else 1

        sog_args: dict = {
            "amp": amp_internal_runtime,
            "bandwidth": bw2_runtime,
            "kernel_param_mode": "internal",
            "kernel_tensor_mode": "external",
            "remove_self_interaction": bool(fitting.remove_self_interaction),
            "use_atomwise": False,
            "use_cubes2_fft": bool(getattr(fitting, "use_cubes2_fft", False)),
            "nlayers": nlayers,
            "trainable_kernel": False,
            "b": float(fitting.b),
        }
        if getattr(fitting, "cubes2_phi_max", None) is not None:
            sog_args["cubes2_phi_max"] = float(fitting.cubes2_phi_max)
        elif getattr(fitting, "n_dl", None) is not None:
            sog_args["n_dl"] = float(fitting.n_dl)
        if getattr(fitting, "charge_neutral_lambda", None) is not None:
            sog_args["charge_neutral_lambda"] = float(fitting.charge_neutral_lambda)
        if getattr(fitting, "charge_neutral", False):
            sog_args["charge_neutral"] = True

        kernel = sog_lib.Sog(
            sog_arguments=sog_args,
            r_cut=float(self.get_rcut()),
        )
        self._sog_direct_kernel_cache[cache_key] = kernel
        return kernel

    # ── Fused SR+LR path ───────────────────────────────────────────────────────────
    def _sog_lr_reduced_energy(
        self,
        base_coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        nloc: int,
    ) -> torch.Tensor:
        """Reciprocal-space (direct k-sum) LR energy, reduced per frame → [nf, 1], with
        the FULL autograd graph intact: positions via ``base_coord``, charges via
        ``latent_charge`` (→ descriptor), box via the passed ``box`` leaf. A downstream
        grad w.r.t. ``base_coord`` therefore captures ∂E_lr/∂r AND the charge-response
        ∂E_lr/∂q·∂q/∂r (conservativity); w.r.t. ``box`` it gives the box-strain Ξ_rec.
        This does NOT detach anything — the caller owns the box-leaf decision."""
        runtime_device = base_coord.device
        real_dtype = base_coord.dtype
        latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)
        nf = base_coord.shape[0]
        nq = latent_charge.shape[-1]
        batch = torch.arange(
            nf, device=runtime_device, dtype=torch.int64
        ).repeat_interleave(nloc)
        kernel = self._build_sog_lib_direct_kernel(runtime_device, real_dtype)
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
        return corr.reshape(nf, 1)

    def _forward_lower_sog_fused(
        self,
        atomic_ret: dict[str, torch.Tensor],
        cc_ext: torch.Tensor,
        box_local: torch.Tensor,
        nloc: int,
        input_prec: str,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Fused SR+LR path: ONE combined descriptor backward for the total
        (short-range + long-range charge-response) energy, force, and virial.
        grad is linear: grad(E_sr+E_lr,·)=grad(E_sr,·)+grad(E_lr,·), so a single
        ``autograd.grad`` on ``E_sr_redu + E_lr_redu`` yields the full conservative
        (charge-response) force.

        Conservativity is preserved because ``E_lr_redu`` carries ``latent_charge``'s
        graph back through the descriptor — the charge-response term ∂E_lr/∂q·∂q/∂r
        is captured automatically (never detach latent_charge/cc_ext).
        """
        redu_prec = env.GLOBAL_PT_ENER_FLOAT_PRECISION
        atom_energy = atomic_ret["energy"]
        latent_charge = atomic_ret["latent_charge"]
        nf = atom_energy.shape[0]
        nall = cc_ext.shape[1]

        need_force = self.do_grad_r("energy") or self.do_grad_c("energy")
        need_virial = self.do_grad_c("energy")

        # short-range reduced energy: non-intensive sum over atoms (matches
        # fit_output_to_model_output)
        e_sr_redu = torch.sum(atom_energy.to(redu_prec), dim=-2)  # [nf, 1]

        # box a leaf so the same backward yields Ξ_rec = ∂E_lr/∂h (no extra kernel call)
        box_leaf = box_local
        if need_virial:
            box_leaf = box_local.detach().clone().requires_grad_(True)

        corr_redu = self._sog_lr_reduced_energy(cc_ext, latent_charge, box_leaf, nloc)
        e_tot_redu = e_sr_redu + corr_redu.to(redu_prec)  # [nf, 1]

        model_ret: dict[str, torch.Tensor] = dict(atomic_ret.items())
        model_ret["energy_redu"] = e_tot_redu

        if need_force:
            grad_inputs = [cc_ext]
            if need_virial:
                grad_inputs = [cc_ext, box_leaf]
            grads = torch.autograd.grad(
                [e_tot_redu],
                grad_inputs,
                grad_outputs=[torch.ones_like(e_tot_redu)],
                create_graph=self.training,
                retain_graph=True,
            )
            force_ext = -grads[0]
            assert force_ext is not None
            model_ret["energy_derv_r"] = force_ext.unsqueeze(-2)  # [nf, nall, 1, 3]
            if need_virial:
                box_grad = grads[1]
                assert box_grad is not None
                # per-atom position virial F_a ⊗ r_a (= SR virial + charge-response Ξ_c)
                pos_vir = torch.einsum(
                    "fak,faj->fakj", force_ext, cc_ext
                ).reshape(nf, nall, 1, 9)
                if do_atomic_virial:
                    pos_vir = pos_vir + atomic_virial_corr(cc_ext, atom_energy).reshape(
                        nf, nall, 1, 9
                    ).to(pos_vir.dtype)
                # reciprocal box-strain Ξ_rec (global), spread over local atoms so the
                # framework's ghost re-sum recovers the correct global virial
                box_vir_redu = (
                    -torch.einsum("fga,fgb->fab", box_grad, box_leaf)
                ).reshape(nf, 1, 9)
                energy_derv_c = pos_vir.clone()
                energy_derv_c[:, :nloc, :, :] = energy_derv_c[:, :nloc, :, :] + (
                    box_vir_redu / nloc
                ).to(energy_derv_c.device).unsqueeze(1)
                model_ret["energy_derv_c"] = energy_derv_c
                model_ret["energy_derv_c_redu"] = torch.sum(
                    energy_derv_c.to(redu_prec), dim=1
                )
        return self._output_type_cast(model_ret, input_prec)

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

        fitting = self.atomic_model.fitting_net
        external_k = fitting is not None and bool(getattr(fitting, "external_kspace", False))

        # ── External kspace or TorchScript export → SR only (no in-model LR) ──
        if external_k or torch.jit.is_scripting():
            model_ret = fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                cc_ext,
                do_atomic_virial=do_atomic_virial,
                create_graph=self.training,
                mask=atomic_ret.get("mask"),
                extended_coord_corr=extended_coord_corr,
            )
            return self._output_type_cast(model_ret, input_prec)

        # ── Validate prerequisites for in-model LR correction ──
        if runtime_box is None:
            raise ValueError(
                "SOG model requires a periodic box for long-range correction. "
                "Set external_kspace=True in the fitting net if LR is provided externally."
            )
        if "latent_charge" not in atomic_ret:
            raise ValueError(
                "SOG model requires latent_charge in atomic_ret for long-range correction. "
                "The fitting net must produce latent_charge."
            )

        # ── Fused SR+LR: single backward for total energy/force/virial ──
        nf_l, nloc_l = nlist.shape[0], nlist.shape[1]
        return self._forward_lower_sog_fused(
            atomic_ret,
            cc_ext,
            runtime_box.view(nf_l, 3, 3),
            nloc_l,
            input_prec,
            do_atomic_virial=do_atomic_virial,
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
        latent_charge = atomic_ret["latent_charge"]  # [nf, nloc, nq]; zero-mean done in forward_common_atomic
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
        # (zero-mean latent_charge now done in forward_common_atomic — single canonical point)
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
        # (zero-mean latent_charge now done in forward_common_atomic — single canonical point)
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
