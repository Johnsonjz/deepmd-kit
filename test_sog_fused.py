"""Verify the fused SR+LR path in SOGEnergyModel:
  (1) conservativity: fused force == -dE/dr (finite difference, incl. charge response)
  (2) amp/bandwidth gradients still flow (no detach/starvation)
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
model.train()  # fused path always active

coord = (torch.rand((nf, nloc, 3), dtype=dtype) * 4.0)
cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 6.0
atype = torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.int64)

# Get baseline results from the fused path
ret = model.forward(coord, atype, box=cell, do_atomic_virial=False)
E = ret["energy"].reshape(()).detach().clone()
F = ret["force"][0].detach().clone()
V = ret["virial"][0].reshape(3, 3).detach().clone()

print("=== (1) conservativity: fused force == -dE/dr (FD) ===")
def energy_redu(c):
    return model.forward(c, atype, box=cell, do_atomic_virial=False)["energy"].reshape(())

Fana = F  # fused analytic force
dx, maxrel = 1e-5, 0.0
for i in (0, 2, 4):
    for d in range(3):
        cp = coord.clone(); cp[0, i, d] += dx
        cm = coord.clone(); cm[0, i, d] -= dx
        ffd = -((energy_redu(cp) - energy_redu(cm)) / (2 * dx)).item()
        rel = abs(Fana[i, d].item() - ffd) / max(abs(ffd), 1e-6)
        maxrel = max(maxrel, rel)
ok1 = maxrel < 1e-4
print(f"  max rel force error = {maxrel:.2e}  -> {'PASS (conservative)' if ok1 else 'FAIL'}")

print("\n=== (2) amp/bandwidth gradients flow (fused path) ===")
model.zero_grad(set_to_none=True)
ret = model.forward(coord, atype, box=cell, do_atomic_virial=False)
loss = ret["energy"].reshape(()) + ret["force"].pow(2).sum() + ret["virial"].pow(2).sum()
loss.backward()
amp_g = model.atomic_model.fitting_net.amp.grad
bw_g = model.atomic_model.fitting_net.bandwidth.grad
amp_ok = amp_g is not None and amp_g.abs().sum().item() > 0
bw_ok = bw_g is not None and bw_g.abs().sum().item() > 0
ok2 = amp_ok and bw_ok
print(f"  amp.grad       nonzero: {amp_ok}  ({None if amp_g is None else amp_g.abs().sum().item():.3e})")
print(f"  bandwidth.grad nonzero: {bw_ok}  ({None if bw_g is None else bw_g.abs().sum().item():.3e})")
print(f"  -> {'PASS' if ok2 else 'FAIL'}")

print(f"\nOVERALL: {'ALL PASS' if (ok1 and ok2) else 'FAIL'}")
