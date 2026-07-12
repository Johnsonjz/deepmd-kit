"""Verify the fused SR+LR training path in SOGEnergyModel:
  (1) fused ≡ two-pass  (force, virial, energy)  to float64 precision
  (2) conservativity: fused force == -dE/dr (finite difference, incl. charge response)
  (3) amp/bandwidth gradients still flow (no detach/starvation)
CPU-only, tiny system — no GPU contention with running training.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_default_device("cpu")

from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.model.sog_model import SOGEnergyModel
from deepmd.pt.model.task.sog_energy_fitting import SOGEnergyFittingNet

dtype = torch.float64
torch.manual_seed(2031)
nf, nloc, nt = 1, 6, 2
rcut, rcut_smth, sel = 4.0, 3.5, [20, 20]

desc = DescrptSeA(rcut, rcut_smth, sel).to(dtype)
fit = SOGEnergyFittingNet(
    var_name="energy", ntypes=nt, dim_descrpt=desc.get_dim_out(),
    dim_out_sr=1, dim_out_lr=1, mixed_types=desc.mixed_types(),
    n_dl=2, external_kspace=False,
).to(dtype)
model = SOGEnergyModel(descriptor=desc, fitting=fit, type_map=["A", "B"]).to(dtype)
model.train()   # fused path is training-gated

coord = (torch.rand((nf, nloc, 3), dtype=dtype) * 4.0)
cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 6.0
atype = torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.int64)


def run(fused):
    model._fused_lr_backward = fused
    ret = model.forward(coord, atype, box=cell, do_atomic_virial=False)
    return (ret["energy"].reshape(()).detach().clone(),
            ret["force"][0].detach().clone(),
            ret["virial"][0].reshape(3, 3).detach().clone())

print("=== (1) fused ≡ two-pass ===")
E_f, F_f, V_f = run(True)
E_t, F_t, V_t = run(False)
dE = (E_f - E_t).abs().item()
dF = (F_f - F_t).abs().max().item()
dV = (V_f - V_t).abs().max().item()
print(f"  |dE|            = {dE:.2e}")
print(f"  max|dF|         = {dF:.2e}")
print(f"  max|dV| (global)= {dV:.2e}")
ok1 = dE < 1e-9 and dF < 1e-9 and dV < 1e-9
print(f"  -> {'PASS' if ok1 else 'FAIL'} (fused matches two-pass)")

print("\n=== (2) conservativity: fused force == -dE/dr (FD) ===")
model._fused_lr_backward = True
def energy_redu(c):
    return model.forward(c, atype, box=cell, do_atomic_virial=False)["energy"].reshape(()).detach()
Fana = F_f  # fused analytic force
dx, maxrel = 1e-5, 0.0
for i in (0, 2, 4):
    for d in range(3):
        cp = coord.clone(); cp[0, i, d] += dx
        cm = coord.clone(); cm[0, i, d] -= dx
        ffd = -((energy_redu(cp) - energy_redu(cm)) / (2 * dx)).item()
        rel = abs(Fana[i, d].item() - ffd) / max(abs(ffd), 1e-6)
        maxrel = max(maxrel, rel)
print(f"  max rel force error = {maxrel:.2e}  -> {'PASS (conservative)' if maxrel < 1e-4 else 'FAIL'}")

print("\n=== (3) amp/bandwidth gradients flow (fused path) ===")
model._fused_lr_backward = True
model.zero_grad(set_to_none=True)
ret = model.forward(coord, atype, box=cell, do_atomic_virial=False)
loss = ret["energy"].reshape(()) + ret["force"].pow(2).sum() + ret["virial"].pow(2).sum()
loss.backward()
amp_g = model.atomic_model.fitting_net.amp.grad
bw_g = model.atomic_model.fitting_net.bandwidth.grad
amp_ok = amp_g is not None and amp_g.abs().sum().item() > 0
bw_ok = bw_g is not None and bw_g.abs().sum().item() > 0
print(f"  amp.grad       nonzero: {amp_ok}  ({None if amp_g is None else amp_g.abs().sum().item():.3e})")
print(f"  bandwidth.grad nonzero: {bw_ok}  ({None if bw_g is None else bw_g.abs().sum().item():.3e})")
print(f"  -> {'PASS' if amp_ok and bw_ok else 'FAIL'}")

print(f"\nOVERALL: {'ALL PASS' if (ok1 and maxrel < 1e-4 and amp_ok and bw_ok) else 'FAIL'}")
