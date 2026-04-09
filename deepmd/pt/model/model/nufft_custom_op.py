# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import warnings

import torch


_DISABLED_OPS: set[str] = set()


class _SOGCorrEnergyAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        amp: torch.Tensor,
        bandwidth: torch.Tensor,
        nk_cpu_tensor: torch.Tensor,
        remove_self_interaction: bool,
    ) -> torch.Tensor:
        nk_cpu = [int(x) for x in nk_cpu_tensor.detach().cpu().tolist()]
        corr = torch.ops.deepmd.nufft_sog_frame_correction_bundle(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            bool(remove_self_interaction),
            False,
            False,
        )[0]
        ctx.save_for_backward(coord, latent_charge, box, amp, bandwidth, nk_cpu_tensor)
        ctx.remove_self_interaction = bool(remove_self_interaction)
        return corr

    @staticmethod
    def backward(ctx, grad_corr: torch.Tensor):
        coord, latent_charge, box, amp, bandwidth, nk_cpu_tensor = ctx.saved_tensors
        nk_cpu = [int(x) for x in nk_cpu_tensor.detach().cpu().tolist()]
        grad_corr = grad_corr.contiguous()
        ret = torch.ops.deepmd.nufft_sog_frame_correction_backward_energy(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            bool(ctx.remove_self_interaction),
            grad_corr,
        )
        grad_latent_charge, grad_amp, grad_bandwidth = ret
        return None, grad_latent_charge, None, grad_amp, grad_bandwidth, None, None


class _SOGCorrForceAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        amp: torch.Tensor,
        bandwidth: torch.Tensor,
        nk_cpu_tensor: torch.Tensor,
        remove_self_interaction: bool,
    ):
        nk_cpu = [int(x) for x in nk_cpu_tensor.detach().cpu().tolist()]
        ret = torch.ops.deepmd.nufft_sog_frame_correction_bundle(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            bool(remove_self_interaction),
            True,
            False,
        )
        corr = ret[0]
        force = ret[1]
        ctx.save_for_backward(coord, latent_charge, box, amp, bandwidth, nk_cpu_tensor, force)
        ctx.remove_self_interaction = bool(remove_self_interaction)
        return corr, force

    @staticmethod
    def backward(ctx, grad_corr: torch.Tensor | None, grad_force: torch.Tensor | None):
        coord, latent_charge, box, amp, bandwidth, nk_cpu_tensor, force = ctx.saved_tensors
        nk_cpu = [int(x) for x in nk_cpu_tensor.detach().cpu().tolist()]
        if grad_corr is None:
            grad_corr = torch.zeros(
                (coord.shape[0], 1),
                dtype=coord.dtype,
                device=coord.device,
            )
        if grad_force is None:
            grad_force = torch.zeros_like(coord)
        grad_corr = grad_corr.contiguous()
        grad_force = grad_force.contiguous()
        need_latent_grad = bool(ctx.needs_input_grad[1])
        ret = torch.ops.deepmd.nufft_sog_frame_correction_backward_bundle(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            bool(ctx.remove_self_interaction),
            grad_corr,
            grad_force,
            force,
            need_latent_grad,
        )
        grad_latent_charge, grad_amp, grad_bandwidth = ret
        return None, grad_latent_charge, None, grad_amp, grad_bandwidth, None, None


class _LESCorrEnergyAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coord: torch.Tensor,
        latent_charge: torch.Tensor,
        box: torch.Tensor,
        sigma: torch.Tensor,
        nk_cpu_tensor: torch.Tensor,
        remove_self_interaction: bool,
    ) -> torch.Tensor:
        nk_cpu = [int(x) for x in nk_cpu_tensor.detach().cpu().tolist()]
        corr = torch.ops.deepmd.nufft_les_frame_correction_bundle(
            coord,
            latent_charge,
            box,
            sigma,
            nk_cpu,
            bool(remove_self_interaction),
            False,
            False,
        )[0]
        ctx.save_for_backward(coord, latent_charge, box, sigma, nk_cpu_tensor)
        ctx.remove_self_interaction = bool(remove_self_interaction)
        return corr

    @staticmethod
    def backward(ctx, grad_corr: torch.Tensor):
        coord, latent_charge, box, sigma, nk_cpu_tensor = ctx.saved_tensors
        nk_cpu = [int(x) for x in nk_cpu_tensor.detach().cpu().tolist()]
        grad_corr = grad_corr.contiguous()
        ret = torch.ops.deepmd.nufft_les_frame_correction_backward_energy(
            coord,
            latent_charge,
            box,
            sigma,
            nk_cpu,
            bool(ctx.remove_self_interaction),
            grad_corr,
        )
        grad_latent_charge, grad_sigma = ret
        return None, grad_latent_charge, None, grad_sigma, None, None


def _warn_once(key: str, message: str) -> None:
    if not hasattr(_warn_once, "_emitted"):
        _warn_once._emitted = set()  # type: ignore[attr-defined]
    emitted = _warn_once._emitted  # type: ignore[attr-defined]
    if key in emitted:
        return
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    emitted.add(key)


def _get_deepmd_op(op_name: str):
    if not hasattr(torch.ops, "deepmd"):
        return None
    if not hasattr(torch.ops.deepmd, op_name):
        return None
    return getattr(torch.ops.deepmd, op_name)


def _deepmd_namespace_loaded() -> bool:
    return hasattr(torch.ops, "deepmd") and hasattr(torch.ops.deepmd, "enable_mpi")


def _raise_op_contract_error(
    feature_name: str,
    missing_ops: list[str],
    *,
    context: str,
) -> None:
    missing = ", ".join(missing_ops)
    raise RuntimeError(
        "DeepMD custom-op namespace is already loaded, but required "
        f"{feature_name} ops are missing: {missing}. "
        f"Context: {context}. "
        "This usually indicates a mismatched or stale libdeepmd_op_pt.so in the runtime environment."
    )


def maybe_sog_frame_correction_bundle(
    coord: torch.Tensor,
    latent_charge: torch.Tensor,
    box: torch.Tensor,
    amp: torch.Tensor,
    bandwidth: torch.Tensor,
    nk_cpu: list[int],
    remove_self_interaction: bool,
    need_force: bool,
    need_virial: bool,
) -> dict[str, torch.Tensor] | None:
    """Try calling customized deepmd op for SOG correction.

    Returns None when op is unavailable or fails at runtime.
    """
    autograd_needed = (
        coord.requires_grad
        or latent_charge.requires_grad
        or box.requires_grad
        or amp.requires_grad
        or bandwidth.requires_grad
    )
    fwd_op = _get_deepmd_op("nufft_sog_frame_correction_bundle")
    bwd_op = _get_deepmd_op("nufft_sog_frame_correction_backward_energy")
    bwd_bundle_op = _get_deepmd_op("nufft_sog_frame_correction_backward_bundle")

    if _deepmd_namespace_loaded():
        missing_ops: list[str] = []
        if fwd_op is None:
            missing_ops.append("nufft_sog_frame_correction_bundle")
        if (
            autograd_needed
            and not need_force
            and not need_virial
            and not coord.requires_grad
            and not box.requires_grad
            and bwd_op is None
        ):
            missing_ops.append("nufft_sog_frame_correction_backward_energy")
        if (
            autograd_needed
            and need_force
            and not need_virial
            and bwd_bundle_op is None
        ):
            missing_ops.append("nufft_sog_frame_correction_backward_bundle")
        if missing_ops:
            _raise_op_contract_error(
                "SOG NUFFT",
                missing_ops,
                context=(
                    "autograd_needed="
                    f"{autograd_needed}, need_force={need_force}, need_virial={need_virial}"
                ),
            )

    if autograd_needed:
        if need_virial:
            return None
        if need_force:
            if fwd_op is None or bwd_bundle_op is None:
                return None
            try:
                nk_cpu_tensor = torch.tensor(
                    list(nk_cpu),
                    dtype=torch.int64,
                    device=coord.device,
                )
                corr, force = _SOGCorrForceAutogradFn.apply(
                    coord,
                    latent_charge,
                    box,
                    amp,
                    bandwidth,
                    nk_cpu_tensor,
                    bool(remove_self_interaction),
                )
                return {"corr_redu": corr, "force_local": force}
            except Exception as exc:  # pragma: no cover - runtime fallback path
                _warn_once(
                    "nufft_sog_frame_correction_bundle_force_autograd_fallback",
                    "Customized SOG NUFFT force autograd path failed, fallback to pytorch_finufft path. "
                    f"Error: {type(exc).__name__}: {exc}",
                )
                return None
        if coord.requires_grad or box.requires_grad:
            return None
        if fwd_op is None or bwd_op is None:
            return None
        try:
            nk_cpu_tensor = torch.tensor(
                list(nk_cpu),
                dtype=torch.int64,
                device=coord.device,
            )
            corr = _SOGCorrEnergyAutogradFn.apply(
                coord,
                latent_charge,
                box,
                amp,
                bandwidth,
                nk_cpu_tensor,
                bool(remove_self_interaction),
            )
            return {"corr_redu": corr}
        except Exception as exc:  # pragma: no cover - runtime fallback path
            _warn_once(
                "nufft_sog_frame_correction_bundle_autograd_fallback",
                "Customized SOG NUFFT autograd path failed, fallback to pytorch_finufft path. "
                f"Error: {type(exc).__name__}: {exc}",
            )
            return None

    if "nufft_sog_frame_correction_bundle" in _DISABLED_OPS:
        return None

    op = fwd_op
    if op is None:
        return None

    try:
        ret = op(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            list(nk_cpu),
            bool(remove_self_interaction),
            bool(need_force),
            bool(need_virial),
        )
    except Exception as exc:  # pragma: no cover - runtime fallback path
        _DISABLED_OPS.add("nufft_sog_frame_correction_bundle")
        _warn_once(
            "nufft_sog_frame_correction_bundle_runtime_fallback",
            "Customized SOG NUFFT op failed at runtime, fallback to pytorch_finufft path. "
            f"Error: {type(exc).__name__}: {exc}",
        )
        return None

    if not isinstance(ret, (tuple, list)) or len(ret) < 1:
        _warn_once(
            "nufft_sog_frame_correction_bundle_bad_return",
            "Customized SOG NUFFT op returned unexpected value. Falling back.",
        )
        return None

    out: dict[str, torch.Tensor] = {"corr_redu": ret[0]}
    if need_force:
        if len(ret) < 2:
            _warn_once(
                "nufft_sog_frame_correction_bundle_missing_force",
                "Customized SOG NUFFT op did not return force tensor. Falling back.",
            )
            return None
        out["force_local"] = ret[1]
    if need_virial:
        if len(ret) < 3:
            _warn_once(
                "nufft_sog_frame_correction_bundle_missing_virial",
                "Customized SOG NUFFT op did not return virial tensor. Falling back.",
            )
            return None
        out["virial_local"] = ret[2]
    return out


def maybe_les_frame_correction_bundle(
    coord: torch.Tensor,
    latent_charge: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    nk_cpu: list[int],
    remove_self_interaction: bool,
    need_force: bool,
    need_virial: bool,
) -> dict[str, torch.Tensor] | None:
    """Try calling customized deepmd op for LES correction.

    Returns None when op is unavailable or fails at runtime.
    """
    autograd_needed = (
        coord.requires_grad
        or latent_charge.requires_grad
        or box.requires_grad
        or sigma.requires_grad
    )
    fwd_op = _get_deepmd_op("nufft_les_frame_correction_bundle")
    bwd_op = _get_deepmd_op("nufft_les_frame_correction_backward_energy")

    if _deepmd_namespace_loaded():
        missing_ops: list[str] = []
        if fwd_op is None:
            missing_ops.append("nufft_les_frame_correction_bundle")
        if (
            autograd_needed
            and not need_force
            and not need_virial
            and not coord.requires_grad
            and not box.requires_grad
            and bwd_op is None
        ):
            missing_ops.append("nufft_les_frame_correction_backward_energy")
        if missing_ops:
            _raise_op_contract_error(
                "LES NUFFT",
                missing_ops,
                context=(
                    "autograd_needed="
                    f"{autograd_needed}, need_force={need_force}, need_virial={need_virial}"
                ),
            )

    if autograd_needed:
        if need_force or need_virial:
            return None
        # Current backward op covers latent_charge/sigma only.
        if coord.requires_grad or box.requires_grad:
            return None
        if fwd_op is None or bwd_op is None:
            return None
        try:
            nk_cpu_tensor = torch.tensor(
                list(nk_cpu),
                dtype=torch.int64,
                device=coord.device,
            )
            corr = _LESCorrEnergyAutogradFn.apply(
                coord,
                latent_charge,
                box,
                sigma,
                nk_cpu_tensor,
                bool(remove_self_interaction),
            )
            return {"corr_redu": corr}
        except Exception as exc:  # pragma: no cover - runtime fallback path
            _warn_once(
                "nufft_les_frame_correction_bundle_autograd_fallback",
                "Customized LES NUFFT autograd path failed, fallback to pytorch_finufft path. "
                f"Error: {type(exc).__name__}: {exc}",
            )
            return None

    if "nufft_les_frame_correction_bundle" in _DISABLED_OPS:
        return None

    op = fwd_op
    if op is None:
        return None

    try:
        ret = op(
            coord,
            latent_charge,
            box,
            sigma,
            list(nk_cpu),
            bool(remove_self_interaction),
            bool(need_force),
            bool(need_virial),
        )
    except Exception as exc:  # pragma: no cover - runtime fallback path
        _DISABLED_OPS.add("nufft_les_frame_correction_bundle")
        _warn_once(
            "nufft_les_frame_correction_bundle_runtime_fallback",
            "Customized LES NUFFT op failed at runtime, fallback to pytorch_finufft path. "
            f"Error: {type(exc).__name__}: {exc}",
        )
        return None

    if not isinstance(ret, (tuple, list)) or len(ret) < 1:
        _warn_once(
            "nufft_les_frame_correction_bundle_bad_return",
            "Customized LES NUFFT op returned unexpected value. Falling back.",
        )
        return None

    out: dict[str, torch.Tensor] = {"corr_redu": ret[0]}
    if need_force:
        if len(ret) < 2:
            _warn_once(
                "nufft_les_frame_correction_bundle_missing_force",
                "Customized LES NUFFT op did not return force tensor. Falling back.",
            )
            return None
        out["force_local"] = ret[1]
    if need_virial:
        if len(ret) < 3:
            _warn_once(
                "nufft_les_frame_correction_bundle_missing_virial",
                "Customized LES NUFFT op did not return virial tensor. Falling back.",
            )
            return None
        out["virial_local"] = ret[2]
    return out
