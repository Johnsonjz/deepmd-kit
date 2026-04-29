# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)
from deepmd.pt.model.task.lr_fitting import (
    LRFittingNet,
)

SOG_DEFAULT_B = to_numpy_array(np.array(1.62976708826776469))
SOG_DEFAULT_SIGMA = to_numpy_array(np.array(2.180230445405648))
SOG_DEFAULT_M = int(12)


@LRFittingNet.register("sog_energy")
@fitting_check_output
class SOGEnergyFittingNet(LRFittingNet):
    """Construct a SOG sr+lr interactions fitting net.

    Parameters
    ----------
    var_name : str
        The atomic property to fit.
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    dim_out_sr : int
        The output dimension of the sr fitting net.
    dim_out_lr : int
        The output dimension of the lr fitting net.
    neuron_sr : list[int]
        Number of neurons in each hidden layers of the sr fitting net.
    neuron_lr : list[int]
        Number of neurons in each hidden layers of the lr fitting net.
    bias_atom_e : torch.Tensor, optional
        Average energy per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
    exclude_types: list[int]
        Atomic contributions of the excluded atom types are set zero.
    trainable : Union[list[bool], bool]
        If the parameters in the fitting net are trainable.
        Now this only supports setting all the parameters in the fitting net at one state.
        When in list[bool], the trainable will be True only if all the boolean parameters are True.
    remove_vaccum_contribution: list[bool], optional
        Remove vacuum contribution before the bias is added. The list assigned each
        type. For `mixed_types` provide `[True]`, otherwise it should be a list of the same
        length as `ntypes` signaling if or not removing the vacuum contribution for the atom types in the list.
    type_map: list[str], Optional
        A list of strings. Give the name to each type of atoms.
    use_aparam_as_mask: bool
        If True, the aparam will not be used in fitting net for embedding.
    default_fparam: list[float], optional
        The default frame parameter. If set, when `fparam.npy` files are not included in the data system,
        this value will be used as the default value for the frame parameter in the fitting net.
    b : float
        Geometric base used by SOG parameterization.
    sigma : float
        Base bandwidth used by SOG parameterization.
    M : int
        Number of geometric bandwidth levels.
    n_dl : float
        NUFFT long-range grid density control factor.
    remove_self_interaction : bool
        If True, remove self interaction term in long-range correction.
    """

    def __init__(
        self,
        var_name: str,
        ntypes: int,
        dim_descrpt: int,
        dim_out_sr: int,
        dim_out_lr: int,
        neuron_sr: list[int] = [128, 128, 128],
        neuron_lr: list[int] = [128, 128, 128],
        bias_atom_e: torch.Tensor | None = None,
        bias_atom_q: torch.Tensor | None = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        rcond: float | None = None,
        seed: int | list[int] | None = None,
        exclude_types: list[int] = [],
        trainable: bool | list[bool] = True,
        remove_vaccum_contribution: list[bool] | None = None,
        type_map: list[str] | None = None,
        use_aparam_as_mask: bool = False,
        default_fparam: list[float] | None = None,
        amp: float | list[float] | torch.Tensor | None = None,
        bandwidth: list[float] | torch.Tensor | None = None,
        b: float | torch.Tensor | None = None,
        sigma: float | torch.Tensor | None = None,
        M: int | None = None,
        n_dl: float | int = 1.0,
        remove_self_interaction: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            var_name=var_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out_sr=dim_out_sr,
            dim_out_lr=dim_out_lr,
            neuron_sr=neuron_sr,
            neuron_lr=neuron_lr,
            bias_atom_e=bias_atom_e,
            bias_atom_q=bias_atom_q,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            rcond=rcond,
            seed=seed,
            exclude_types=exclude_types,
            trainable=trainable,
            remove_vaccum_contribution=remove_vaccum_contribution,
            type_map=type_map,
            use_aparam_as_mask=use_aparam_as_mask,
            default_fparam=default_fparam,
            **kwargs,
        )
        if b is None:
            b_tensor = torch.as_tensor(SOG_DEFAULT_B, dtype=dtype, device=device)
        else:
            b_tensor = torch.as_tensor(b, dtype=dtype, device=device)
        if b_tensor.numel() == 0:
            b_tensor = torch.as_tensor(SOG_DEFAULT_B, dtype=dtype, device=device)
        b_value = float(b_tensor.reshape(-1)[0].item())
        if b_value <= 0.0:
            raise ValueError("`b` should be positive.")

        if sigma is None:
            sigma_tensor = torch.as_tensor(SOG_DEFAULT_SIGMA, dtype=dtype, device=device)
        else:
            sigma_tensor = torch.as_tensor(sigma, dtype=dtype, device=device)
        if sigma_tensor.numel() == 0:
            sigma_tensor = torch.as_tensor(SOG_DEFAULT_SIGMA, dtype=dtype, device=device)
        sigma_value = float(sigma_tensor.reshape(-1)[0].item())
        if sigma_value <= 0.0:
            raise ValueError("`sigma` should be positive.")

        m_value = SOG_DEFAULT_M if M is None else int(M)
        m_value = max(1, m_value)

        if bandwidth is None:
            b_base = torch.tensor(b_value, dtype=dtype, device=device)
            bw_tensor = sigma_value * torch.pow(
                b_base,
                torch.arange(m_value, dtype=dtype, device=device),
            )
            # Keep bandwidth as bw^2 so the kernel uses
            # amp_m * bandwidth_m * exp(-0.5 * bandwidth_m * k^2).
            bandwidth_tensor = bw_tensor.square()
        else:
            bandwidth_tensor = torch.as_tensor(bandwidth, dtype=dtype, device=device).reshape(-1)
        if bandwidth_tensor.numel() == 0:
            raise ValueError("`bandwidth` should not be empty.")
        if not torch.isfinite(bandwidth_tensor).all():
            raise ValueError("`bandwidth` should be finite.")
        if torch.any(bandwidth_tensor <= 0.0):
            raise ValueError("`bandwidth` values should be positive.")

        if amp is None:
            coef1 = float(4.0 * np.pi * np.log(b_value))
            amp_tensor = torch.full_like(bandwidth_tensor, coef1)
        else:
            amp_tensor = torch.as_tensor(amp, dtype=dtype, device=device).reshape(-1)
        if amp_tensor.numel() == 0:
            raise ValueError("`amp` should not be empty.")
        if not torch.isfinite(amp_tensor).all():
            raise ValueError("`amp` should be finite.")

        if amp_tensor.numel() == 1 and bandwidth_tensor.numel() > 1:
            amp_tensor = amp_tensor.expand_as(bandwidth_tensor).clone()
        elif amp_tensor.numel() != bandwidth_tensor.numel():
            raise ValueError(
                "`amp` should be scalar or have the same length as `bandwidth`."
            )

        n_dl_value = float(n_dl)
        if (not np.isfinite(n_dl_value)) or n_dl_value <= 0.0:
            raise ValueError("`n_dl` should be a positive finite number.")

        self.n_dl = n_dl_value
        self.amp = torch.nn.Parameter(
            amp_tensor,
            requires_grad=bool(self.trainable),
        )
        self.bandwidth = torch.nn.Parameter(
            bandwidth_tensor,
            requires_grad=bool(self.trainable),
        )
        self.b = b_value
        self.sigma = sigma_value
        self.M = m_value
        self.remove_self_interaction = bool(remove_self_interaction)
        self._nufft_fallback_warned = False

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    name="latent_charge",
                    shape=[self.dim_out_lr],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                ),
            ]
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["type"] = "sog_energy"
        variables = data["@variables"]
        variables["amp"] = to_numpy_array(self.amp)
        variables["bandwidth"] = to_numpy_array(self.bandwidth)
        data["b"] = float(self.b)
        data["sigma"] = float(self.sigma)
        data["M"] = int(self.M)
        data["n_dl"] = self.n_dl
        data["remove_self_interaction"] = bool(self.remove_self_interaction)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "SOGEnergyFittingNet":
        data = data.copy()

        variables = data.get("@variables", {}).copy()

        amp_tensor = to_torch_tensor(variables.pop("amp", None))
        bandwidth_tensor = to_torch_tensor(variables.pop("bandwidth", None))
        data["@variables"] = variables

        obj = super().deserialize(data)

        with torch.no_grad():
            if bandwidth_tensor is not None:
                bw = bandwidth_tensor.to(
                    dtype=obj.bandwidth.dtype,
                    device=obj.bandwidth.device,
                ).reshape(-1)
                if bw.numel() == 0:
                    raise ValueError("`bandwidth` should not be empty.")
                if not torch.isfinite(bw).all():
                    raise ValueError("`bandwidth` should be finite.")
                if torch.any(bw <= 0.0):
                    raise ValueError("`bandwidth` values should be positive.")

                if obj.bandwidth.shape != bw.shape:
                    obj.bandwidth = torch.nn.Parameter(
                        bw,
                        requires_grad=bool(obj.trainable),
                    )
                else:
                    obj.bandwidth.copy_(bw)
                obj.M = int(obj.bandwidth.numel())

            if amp_tensor is not None:
                amp_new = amp_tensor.to(dtype=obj.amp.dtype, device=obj.amp.device).reshape(-1)
                if amp_new.numel() == 0:
                    raise ValueError("`amp` should not be empty.")
                if not torch.isfinite(amp_new).all():
                    raise ValueError("`amp` should be finite.")

                if amp_new.numel() == 1 and obj.bandwidth.numel() > 1:
                    amp_new = amp_new.expand_as(obj.bandwidth).clone()
                elif amp_new.numel() != obj.bandwidth.numel():
                    raise ValueError(
                        "`amp` should be scalar or have the same length as `bandwidth`."
                    )

                if obj.amp.shape != amp_new.shape:
                    obj.amp = torch.nn.Parameter(
                        amp_new,
                        requires_grad=bool(obj.trainable),
                    )
                else:
                    obj.amp.copy_(amp_new)
            elif obj.amp.numel() == 1 and obj.bandwidth.numel() > 1:
                obj.amp = torch.nn.Parameter(
                    obj.amp.detach().expand_as(obj.bandwidth).clone(),
                    requires_grad=bool(obj.trainable),
                )
        return obj

    def _kernel_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.amp, self.bandwidth

    def forward(
        self,
        descriptor: torch.Tensor,
        atype: torch.Tensor,
        gr: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        h2: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self._forward_common(
            descriptor=descriptor,
            atype=atype,
            gr=gr,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        result = {
            "energy": out["sr"],
            "latent_charge": out["lr"],
        }
        if "middle_output" in out:
            result["middle_output"] = out["middle_output"]
        return result

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
