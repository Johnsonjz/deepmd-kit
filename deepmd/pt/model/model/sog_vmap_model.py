# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from typing import (
    Any,
)

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

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._kgrid_base_cache: dict = {}

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

    def _sog_lr_reduced_energy(
        self,
        base_coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        nloc: int,
    ) -> torch.Tensor:
        """NUFFT-based reciprocal-space LR energy, reduced per frame -> [nf, 1].
        Full autograd graph intact — the fused path's ``autograd.grad`` backprops
        through FINUFFT to capture both SR and LR force/virial in a single backward.
        """
        runtime_device = base_coord.device
        real_dtype = base_coord.dtype
        complex_dtype = (
            torch.complex128 if real_dtype == torch.float64 else torch.complex64
        )

        coord = base_coord[:, :nloc, :]
        latent_charge = latent_charge[:, :nloc, :].to(device=runtime_device, dtype=real_dtype)
        box = box.to(device=runtime_device, dtype=real_dtype)

        if box.dim() != 3 or box.shape[-2:] != (3, 3):
            raise ValueError(
                f"`box` should be [nf, 3, 3], got shape {tuple(box.shape)}"
            )

        fitting = self.get_fitting_net()
        remove_self_interaction = bool(fitting.remove_self_interaction)
        amp = torch.as_tensor(
            fitting.amp, dtype=real_dtype, device=runtime_device
        ).reshape(-1)
        bandwidth = torch.as_tensor(
            fitting.bandwidth, dtype=real_dtype, device=runtime_device
        )

        n_dl = float(fitting.n_dl)
        pi_tensor = torch.tensor(torch.pi, dtype=real_dtype, device=runtime_device)
        two_pi = 2.0 * pi_tensor
        n_dl_tensor = torch.as_tensor(n_dl, dtype=real_dtype, device=runtime_device)
        k_sq_max = (two_pi / n_dl_tensor) ** 2
        coulomb_to_ev = torch.as_tensor(
            E2_PER_ANGSTROM_TO_EV, dtype=real_dtype, device=runtime_device
        )

        nf = coord.shape[0]
        corr = torch.zeros((nf, 1), dtype=real_dtype, device=runtime_device)

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
            tuple(max(1, int(v.item() / n_dl)) for v in norms_all[ff])
            for ff in range(nf)
        ]
        frame_groups: dict[tuple[int, int, int], list[int]] = {}
        for ff, nk in enumerate(nk_per_frame):
            frame_groups.setdefault(nk, []).append(ff)

        bw2 = bandwidth.square().view(1, 1, 1, -1)
        amp = amp.view(1, 1, 1, -1)

        for nk, frame_ids in frame_groups.items():
            k_grid_int, zero_mask, output_shape = self._get_cached_kgrid_base(
                nk, runtime_device, real_dtype
            )
            zero_mask_expand = zero_mask.unsqueeze(0)

            cell_inv_group = cell_inv_all[frame_ids]
            g_cart_group = two_pi * torch.einsum(
                "bik,k...->bi...", cell_inv_group, k_grid_int
            )
            k_sq_group = torch.sum(g_cart_group**2, dim=1)
            k_in_cutoff = k_sq_group <= k_sq_max

            kfac_group = amp * torch.exp(-0.5 * bw2 * k_sq_group.unsqueeze(-1))
            kfac_group = kfac_group.sum(dim=-1).masked_fill(
                zero_mask_expand | (~k_in_cutoff), 0.0
            )

            for local_idx, ff in enumerate(frame_ids):
                q_t = q_all[ff]
                volume = volume_all[ff]
                nufft_points = nufft_points_all[ff]
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
                # FINUFFT coefficients are in FFT order; align to centered mode ordering.
                recon = torch.fft.fftshift(recon, dim=(1, 2, 3))

                rho_sq = recon.real.square() + recon.imag.square()
                corr[ff, 0] = (kfac.unsqueeze(0) * rho_sq).sum() / (2.0 * volume)

                if remove_self_interaction:
                    diag_sum = kfac.sum() / (2.0 * volume)
                    corr[ff, 0] -= torch.sum(latent_charge[ff] ** 2) * diag_sum

        corr = corr * coulomb_to_ev
        return corr  # [nf, 1]
