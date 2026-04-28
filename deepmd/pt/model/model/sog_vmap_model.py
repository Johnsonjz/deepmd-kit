# SPDX-License-Identifier: LGPL-3.0-or-later
import math

import pytorch_finufft
import torch

from deepmd.pt.model.model.model import (
    BaseModel,
)

from .sog_model import (
    E2_PER_ANGSTROM_TO_EV,
    SOGEnergyModel,
)


@BaseModel.register("sog_vmap")
class SOGVmapModel(SOGEnergyModel):
    model_type = "sog_vmap"

    @staticmethod
    def _finufft_type1_shifted_single(
        nufft_points_single: torch.Tensor,
        charge_single: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        recon_single = pytorch_finufft.functional.finufft_type1(
            nufft_points_single,
            charge_single,
            output_shape=output_shape,
            eps=1e-4,
            isign=-1,
        )
        # FINUFFT coefficients are returned in FFT order; align to centered
        # mode ordering (-nk..nk) used by k_grid_int/kfac/g_cart.
        return torch.fft.fftshift(recon_single, dim=(1, 2, 3))

    @staticmethod
    def _finufft_type2_single(
        nufft_points_single: torch.Tensor,
        grad_conv_single: torch.Tensor,
    ) -> torch.Tensor:
        return pytorch_finufft.functional.finufft_type2(
            nufft_points_single,
            grad_conv_single,
            eps=1e-4,
            isign=1,
        )

    def _compute_sog_frame_correction_bundle(
        self,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        *,
        need_force: bool,
        need_virial: bool,
    ) -> dict[str, torch.Tensor]:
        if coord.dim() != 3:
            raise ValueError(
                f"`coord` should be [nf, nloc, 3], got shape {tuple(coord.shape)}"
            )
        if latent_charge.dim() != 3:
            raise ValueError(
                f"`latent_charge` should be [nf, nloc, nq], got shape {tuple(latent_charge.shape)}"
            )
        if coord.shape[:2] != latent_charge.shape[:2]:
            raise ValueError(
                "`coord` and `latent_charge` local dimensions mismatch: "
                f"{tuple(coord.shape[:2])} vs {tuple(latent_charge.shape[:2])}"
            )

        fitting = self.get_fitting_net()
        runtime_device = coord.device
        real_dtype = coord.dtype
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )
        latent_charge = latent_charge.to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)
        if box.dim() != 3 or box.shape[-2:] != (3, 3):
            raise ValueError(
                f"`box` should be [nf, 3, 3], got shape {tuple(box.shape)}"
            )

        remove_self_interaction = bool(fitting.remove_self_interaction)
        amp = torch.as_tensor(
            fitting.amp,
            dtype=real_dtype,
            device=runtime_device,
        ).reshape(-1)
        bandwidth = torch.as_tensor(
            fitting.bandwidth,
            dtype=real_dtype,
            device=runtime_device,
        )
        if amp.numel() == 0:
            raise ValueError("Invalid SOG `amp` value in fitting net.")
        if not torch.isfinite(amp).all():
            raise ValueError("Invalid SOG `amp` value in fitting net.")
        if bandwidth.ndim != 1 or bandwidth.numel() == 0:
            raise ValueError("Invalid SOG `bandwidth` in fitting net.")
        if not torch.isfinite(bandwidth).all():
            raise ValueError("Invalid SOG `bandwidth` in fitting net.")
        if torch.any(bandwidth <= 0.0):
            raise ValueError("SOG `bandwidth` should be positive.")

        if amp.numel() == 1 and bandwidth.numel() > 1:
            amp = amp.expand_as(bandwidth)
        elif amp.numel() != bandwidth.numel():
            raise ValueError(
                "SOG `amp` should be scalar or have the same length as `bandwidth`."
            )
        n_dl = float(fitting.n_dl)
        if (not math.isfinite(n_dl)) or n_dl <= 0.0:
            raise ValueError("`n_dl` should be a positive finite number.")
        pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
        two_pi = 2.0 * pi_tensor
        n_dl_tensor = torch.as_tensor(n_dl, dtype=real_dtype, device=runtime_device)
        k_sq_max = (two_pi / n_dl_tensor) ** 2
        coulomb_to_ev = torch.as_tensor(
            E2_PER_ANGSTROM_TO_EV,
            dtype=real_dtype,
            device=runtime_device,
        )

        nf, nloc, _ = coord.shape
        corr = torch.zeros((nf, 1), dtype=real_dtype, device=runtime_device)
        force_local = (
            torch.zeros((nf, nloc, 3), dtype=real_dtype, device=runtime_device)
            if need_force
            else None
        )
        virial_local = (
            torch.zeros((nf, nloc, 1, 9), dtype=real_dtype, device=runtime_device)
            if need_virial
            else None
        )

        volume_all = torch.det(box)
        if torch.any(torch.abs(volume_all) <= torch.finfo(real_dtype).eps):
            raise ValueError("`box` is singular (near-zero volume), cannot run NUFFT.")

        cell_inv_all = torch.linalg.inv(box)
        r_frac_all = torch.matmul(coord, cell_inv_all)
        r_frac_all = torch.remainder(r_frac_all + 0.5, 1.0) - 0.5
        point_limit = pi_tensor - 32.0 * torch.finfo(real_dtype).eps
        r_in_all = torch.clamp(
            2.0 * pi_tensor * r_frac_all,
            min=-point_limit,
            max=point_limit,
        ).contiguous()
        nufft_points_all = r_in_all.transpose(1, 2).contiguous()
        q_all = latent_charge.transpose(1, 2).contiguous()

        norms_all = torch.norm(box, dim=2)
        nk_per_frame = [
            tuple(max(1, int(v.item() / n_dl)) for v in norms_all[ff]) for ff in range(nf)
        ]
        frame_groups: dict[tuple[int, int, int], list[int]] = {}
        for ff, nk in enumerate(nk_per_frame):
            frame_groups.setdefault(nk, []).append(ff)

        vmap_op = getattr(torch, "vmap", None)
        can_use_vmap = (vmap_op is not None) and (not torch.jit.is_scripting())
        bw2 = bandwidth.square().view(1, 1, 1, -1)
        amp = amp.view(1, 1, 1, -1)
        for nk, frame_ids in frame_groups.items():
            k_grid_int, zero_mask, output_shape = self._get_cached_kgrid_base(
                nk,
                runtime_device,
                real_dtype,
            )
            zero_mask_expand = zero_mask.unsqueeze(0)

            cell_inv_group = cell_inv_all[frame_ids]
            g_cart_group = two_pi * torch.einsum("bik,k...->bi...", cell_inv_group, k_grid_int)
            k_sq_group = torch.sum(g_cart_group**2, dim=1)
            k_in_cutoff = k_sq_group <= k_sq_max

            kfac_group = amp * torch.exp(-0.5 * bw2 * k_sq_group.unsqueeze(-1))
            kfac_group = kfac_group.sum(dim=-1).masked_fill(
                zero_mask_expand | (~k_in_cutoff), 0.0
            )

            if can_use_vmap and len(frame_ids) > 1:
                coord_group = coord[frame_ids]
                q_t_group = q_all[frame_ids].contiguous()
                volume_group = volume_all[frame_ids]
                nufft_points_group = nufft_points_all[frame_ids].contiguous()

                charge_group = (
                    torch.complex(q_t_group, torch.zeros_like(q_t_group))
                    .to(dtype=complex_dtype)
                    .contiguous()
                )

                recon_group = vmap_op(
                    self._finufft_type1_shifted_single,
                    in_dims=(0, 0, None),
                    out_dims=0,
                )(
                    nufft_points_group,
                    charge_group,
                    output_shape,
                )

                rho_sq_group = recon_group.real.square() + recon_group.imag.square()
                corr_group = (
                    (kfac_group.unsqueeze(1) * rho_sq_group).sum(dim=(1, 2, 3, 4))
                    / (2.0 * volume_group)
                )
                corr[frame_ids, 0] = corr_group

                if need_force:
                    conv_group = kfac_group.unsqueeze(1).to(dtype=complex_dtype) * recon_group
                    grad_conv_group = (
                        1j * g_cart_group.unsqueeze(2).to(dtype=complex_dtype)
                    ) * conv_group.unsqueeze(1)
                    # Convert back to FINUFFT FFT order before type-2 evaluation.
                    grad_conv_group = torch.fft.ifftshift(grad_conv_group, dim=(3, 4, 5))

                    grad_field_group = vmap_op(
                        self._finufft_type2_single,
                        in_dims=(0, 0),
                        out_dims=0,
                    )(
                        nufft_points_group,
                        grad_conv_group,
                    )

                    force_group = (
                        -(q_t_group.unsqueeze(1) * grad_field_group.real.to(dtype=real_dtype))
                        .sum(dim=2)
                        .transpose(1, 2)
                    )
                    force_group = force_group / volume_group.view(-1, 1, 1)
                    force_local[frame_ids] = force_group

                    if need_virial:
                        virial_local[frame_ids] = torch.einsum(
                            "bai,baj->baij",
                            force_group,
                            coord_group,
                        ).reshape(len(frame_ids), nloc, 1, 9)

                if remove_self_interaction:
                    diag_sum_group = kfac_group.sum(dim=(1, 2, 3)) / (2.0 * volume_group)
                    corr[frame_ids, 0] -= (
                        latent_charge[frame_ids].square().sum(dim=(1, 2)) * diag_sum_group
                    )
            else:
                for local_idx, ff in enumerate(frame_ids):
                    r_raw = coord[ff]
                    q_t = q_all[ff]
                    volume = volume_all[ff]
                    nufft_points = nufft_points_all[ff]
                    g_cart = g_cart_group[local_idx]
                    kfac = kfac_group[local_idx]

                    charge = (
                        torch.complex(q_t, torch.zeros_like(q_t))
                        .to(dtype=complex_dtype)
                        .contiguous()
                    )
                    recon = pytorch_finufft.functional.finufft_type1(
                        nufft_points,
                        charge,
                        output_shape=output_shape,
                        eps=1e-4,
                        isign=-1,
                    )
                    # FINUFFT coefficients are returned in FFT order; align to centered
                    # mode ordering (-nk..nk) used by k_grid_int/kfac/g_cart.
                    recon = torch.fft.fftshift(recon, dim=(1, 2, 3))

                    rho_sq = recon.real.square() + recon.imag.square()
                    corr[ff, 0] = (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

                    conv = None
                    if need_force:
                        conv = kfac.unsqueeze(0).to(dtype=complex_dtype) * recon

                    if need_force:
                        assert conv is not None
                        grad_conv = (
                            1j * g_cart.unsqueeze(1).to(dtype=complex_dtype)
                        ) * conv.unsqueeze(0)
                        # Convert back to FINUFFT FFT order before type-2 evaluation.
                        grad_conv = torch.fft.ifftshift(grad_conv, dim=(2, 3, 4))
                        grad_field = pytorch_finufft.functional.finufft_type2(
                            nufft_points,
                            grad_conv,
                            eps=1e-4,
                            isign=1,
                        )
                        force_frame = (
                            -(q_t.unsqueeze(0) * grad_field.real.to(dtype=real_dtype))
                            .sum(dim=1)
                            .transpose(0, 1)
                        )
                        force_frame = force_frame / volume
                        force_local[ff] = force_frame

                        if need_virial:
                            virial_local[ff] = torch.einsum(
                                "ai,aj->aij",
                                force_frame,
                                r_raw,
                            ).reshape(nloc, 1, 9)

                    if remove_self_interaction:
                        diag_sum = kfac.sum() / (2.0 * volume)
                        corr[ff, 0] -= torch.sum(latent_charge[ff] ** 2) * diag_sum

        # Convert electrostatic unit from e^2/A to eV.
        corr = corr * coulomb_to_ev
        if force_local is not None:
            force_local = force_local * coulomb_to_ev
        if virial_local is not None:
            virial_local = virial_local * coulomb_to_ev

        out: dict[str, torch.Tensor] = {"corr_redu": corr}
        if force_local is not None:
            out["force_local"] = force_local
        if virial_local is not None:
            out["virial_local"] = virial_local
        return out
