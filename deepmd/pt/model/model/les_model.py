# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import math
import torch

from deepmd.pt.model.atomic_model import (
    LESEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
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

LESEnergyModel_ = make_model(LESEnergyAtomicModel)


@BaseModel.register("les_ener")
class LESEnergyModel(DPModelCommon, LESEnergyModel_):
    model_type = "les_ener"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        LESEnergyModel_.__init__(self, *args, **kwargs)
        self._hessian_enabled = False
        self._kgrid_base_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]
        ] = {}

    @staticmethod
    def _device_key(device: torch.device) -> str:
        if device.index is None:
            return device.type
        return f"{device.type}:{device.index}"

    @staticmethod
    def _trim_cache(cache: dict[Any, Any], max_size: int = 8) -> None:
        if len(cache) > max_size:
            oldest_key = next(iter(cache.keys()))
            cache.pop(oldest_key, None)

    def _get_cached_kgrid_base(
        self,
        nk: tuple[int, int, int],
        runtime_device: torch.device,
        real_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
        cache_key = (
            self._device_key(runtime_device),
            str(real_dtype),
            int(nk[0]),
            int(nk[1]),
            int(nk[2]),
        )
        cached = self._kgrid_base_cache.get(cache_key)
        if cached is not None:
            return cached

        n1 = torch.arange(-nk[0], nk[0] + 1, device=runtime_device, dtype=real_dtype)
        n2 = torch.arange(-nk[1], nk[1] + 1, device=runtime_device, dtype=real_dtype)
        n3 = torch.arange(-nk[2], nk[2] + 1, device=runtime_device, dtype=real_dtype)
        kx_grid, ky_grid, kz_grid = torch.meshgrid(n1, n2, n3, indexing="ij")
        k_grid_int = torch.stack((kx_grid, ky_grid, kz_grid), dim=0)
        zero_mask = (k_grid_int[0] == 0) & (k_grid_int[1] == 0) & (k_grid_int[2] == 0)
        output_shape = tuple(int(x) for x in kx_grid.shape)

        out = (k_grid_int, zero_mask, output_shape)
        self._kgrid_base_cache[cache_key] = out
        self._trim_cache(self._kgrid_base_cache)
        return out

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
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if self._hessian_enabled:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def

    def _compute_les_frame_correction_bundle(
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

        sigma_raw = getattr(fitting, "sigma", None)
        if sigma_raw is None:
            raise ValueError(
                "LES fitting net should provide `sigma` for frame correction."
            )
        sigma = torch.as_tensor(
            sigma_raw,
            dtype=real_dtype,
            device=runtime_device,
        ).reshape(-1)[0]
        sigma = torch.clamp(sigma, min=torch.finfo(real_dtype).eps)
        remove_self_interaction = bool(fitting.remove_self_interaction)
        n_dl = float(fitting.n_dl)
        if (not math.isfinite(n_dl)) or n_dl <= 0.0:
            raise ValueError("`n_dl` should be a positive finite number.")
        two_pi = torch.tensor(2.0 * torch.pi, dtype=real_dtype, device=runtime_device)
        n_dl_tensor = torch.as_tensor(n_dl, dtype=real_dtype, device=runtime_device)
        k_sq_max = (two_pi / n_dl_tensor) ** 2

        nf, nloc, _ = coord.shape
        sqrt_two_pi = torch.sqrt(two_pi)
        coulomb_to_ev = torch.as_tensor(
            E2_PER_ANGSTROM_TO_EV,
            dtype=real_dtype,
            device=runtime_device,
        )

        def _corr_redu(
            positions: torch.Tensor,
            charges: torch.Tensor,
        ) -> torch.Tensor:
            volume_all = torch.det(box)
            if torch.any(torch.abs(volume_all) <= torch.finfo(real_dtype).eps):
                raise ValueError("`box` is singular (near-zero volume).")

            cell_inv_all = torch.linalg.inv(box)
            norms_all = torch.norm(box, dim=2)
            nk_per_frame = [
                tuple(max(1, int(v.item() / n_dl)) for v in norms_all[ff])
                for ff in range(nf)
            ]
            frame_groups: dict[tuple[int, int, int], list[int]] = {}
            for ff, nk in enumerate(nk_per_frame):
                frame_groups.setdefault(nk, []).append(ff)

            corr = torch.zeros((nf, 1), dtype=real_dtype, device=runtime_device)
            for nk, frame_ids in frame_groups.items():
                k_grid_int, zero_mask, _ = self._get_cached_kgrid_base(
                    nk,
                    runtime_device,
                    real_dtype,
                )
                zero_mask_expand = zero_mask.unsqueeze(0)

                cell_inv_group = cell_inv_all[frame_ids]
                g_cart_group = two_pi * torch.einsum(
                    "bik,k...->bi...",
                    cell_inv_group,
                    k_grid_int,
                )
                k_sq_group = torch.sum(g_cart_group**2, dim=1)
                k_mask_group = (~zero_mask_expand) & (k_sq_group <= k_sq_max)
                k_sq_safe_group = torch.where(
                    k_mask_group,
                    k_sq_group,
                    torch.ones_like(k_sq_group),
                )
                kfac_group = (
                    torch.exp(-0.5 * (sigma**2) * k_sq_safe_group) / k_sq_safe_group
                ).masked_fill(~k_mask_group, 0.0)

                for local_idx, ff in enumerate(frame_ids):
                    r_raw = positions[ff]
                    q_now = charges[ff]
                    volume = volume_all[ff]

                    g_cart = g_cart_group[local_idx].reshape(3, -1).transpose(0, 1)
                    kfac_flat = kfac_group[local_idx].reshape(-1)
                    k_mask_flat = kfac_flat != 0.0

                    if torch.any(k_mask_flat):
                        g_cart = g_cart[k_mask_flat]
                        kfac_flat = kfac_flat[k_mask_flat]

                        k_dot_r = torch.matmul(r_raw, g_cart.transpose(0, 1))
                        cos_k_dot_r = torch.cos(k_dot_r)
                        sin_k_dot_r = torch.sin(k_dot_r)

                        s_real = (q_now.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
                        s_imag = (q_now.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
                        rho_sq = s_real.square() + s_imag.square()
                        corr_frame = (kfac_flat.unsqueeze(0) * rho_sq).sum() * two_pi / volume
                    else:
                        corr_frame = torch.zeros((), dtype=real_dtype, device=runtime_device)

                    if remove_self_interaction:
                        corr_frame = corr_frame - torch.sum(q_now**2) / (sigma * sqrt_two_pi)

                    corr[ff, 0] = corr_frame

            return corr * coulomb_to_ev

        if need_force or need_virial:
            coord_for_grad = (
                coord
                if coord.requires_grad
                else coord.detach().clone().requires_grad_(True)
            )
            corr_redu = _corr_redu(coord_for_grad, latent_charge)
            force_local = -torch.autograd.grad(
                [corr_redu],
                [coord_for_grad],
                grad_outputs=[torch.ones_like(corr_redu)],
                create_graph=self.training,
                retain_graph=True,
            )[0]

            out: dict[str, torch.Tensor] = {
                "corr_redu": corr_redu,
                "force_local": force_local,
            }
            if need_virial:
                out["virial_local"] = torch.einsum(
                    "bai,baj->baij",
                    force_local,
                    coord,
                ).reshape(nf, nloc, 1, 9)
            return out

        return {"corr_redu": _corr_redu(coord, latent_charge)}

    def _compute_les_frame_correction(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
    ) -> torch.Tensor:
        out = self._compute_les_frame_correction_bundle(
            coord,
            latent_charge,
            box,
            need_force=False,
            need_virial=False,
        )
        return out["corr_redu"]

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
        
        corr_bundle = self._compute_les_frame_correction_bundle(
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

        if need_force:
            corr_force_local = corr_bundle["force_local"].to(coord_local.dtype)

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
                corr_virial_local = corr_bundle["virial_local"].to(
                    corr_force_local.dtype
                )
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
        model_ret = super().forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=extra_nlist_sort,
            extended_coord_corr=extended_coord_corr,
        )
        runtime_box = box
        if runtime_box is None and comm_dict is not None and "box" in comm_dict:
            runtime_box = comm_dict["box"]
        return self._apply_frame_correction_lower(
            model_ret,
            extended_coord,
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
            if "latent_charge" in model_ret:
                model_predict["latent_charge"] = model_ret["latent_charge"]
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -3
                    )
            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            if self._hessian_enabled:
                model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-2)
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
            if "latent_charge" in model_ret:
                model_predict["latent_charge"] = model_ret["latent_charge"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict
