"""Phase-1 conservativeness check: with the base-grad fix, the in-model SOG force
must equal -dE/dr (finite difference), i.e. include the charge-response term."""
import torch
torch.set_default_device("cpu")

from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.model.sog_model import SOGEnergyModel
from deepmd.pt.model.task.sog_energy_fitting import SOGEnergyFittingNet
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

dtype = torch.float64
torch.manual_seed(2031)

nf, nloc, nt = 1, 6, 2
rcut, rcut_smth, sel = 4.0, 3.5, [20, 20]

desc = DescrptSeA(rcut, rcut_smth, sel).to(dtype)
fit = SOGEnergyFittingNet(
    var_name="energy", ntypes=nt, dim_descrpt=desc.get_dim_out(),
    dim_out_sr=1, dim_out_lr=1, mixed_types=desc.mixed_types(),
    n_dl=2, external_kspace=False,   # in-model LR correction ACTIVE (direct k-sum)
).to(dtype)
model = SOGEnergyModel(descriptor=desc, fitting=fit, type_map=["A", "B"]).to(dtype)
model.eval()

coord = (torch.rand((nf, nloc, 3), dtype=dtype) * 4.0)
cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 6.0
atype = torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.int64)


def energy_redu(c):
    ret = model.forward(c, atype, box=cell, do_atomic_virial=False)
    return ret["energy"].reshape(()).clone()


# analytic force (with the fix, includes charge response)
ret = model.forward(coord, atype, box=cell, do_atomic_virial=False)
print("forward keys:", list(ret.keys()))
assert "force" in ret, f"no force key; keys={list(ret.keys())}"
F = ret["force"][0].detach()  # [nloc, 3]

dx = 1e-5
print(f"{'atom':>4} {'dir':>3} {'F_analytic':>14} {'F_fd(-dE/dr)':>14} {'|diff|':>11} {'rel':>9}")
maxrel = 0.0
for i in (0, 2, 4):
    for d in range(3):
        cp = coord.clone(); cp[0, i, d] += dx
        cm = coord.clone(); cm[0, i, d] -= dx
        ffd = -((energy_redu(cp) - energy_redu(cm)) / (2 * dx)).item()
        fa = F[i, d].item()
        rel = abs(fa - ffd) / max(abs(ffd), 1e-6)
        maxrel = max(maxrel, rel)
        print(f"{i:>4} {d:>3} {fa:>14.6e} {ffd:>14.6e} {abs(fa-ffd):>11.2e} {rel:>9.2e}")
print(f"\nmax rel force error = {maxrel:.2e}  ->  {'PASS (conservative)' if maxrel < 1e-4 else 'FAIL'}")

# ── Virial check: trace(Ξ) must equal -dE/dλ under an isotropic strain (coord & box
#    scaled by 1+λ), which tests the reciprocal box-strain term (DPLR Ξ_rec/Ξ_c). ──
vir = ret["virial"][0].reshape(3, 3).detach()
tr_model = (vir[0, 0] + vir[1, 1] + vir[2, 2]).item()


def energy_strained(lam):
    c = coord * (1.0 + lam)
    b = cell * (1.0 + lam)
    r = model.forward(c, atype, box=b, do_atomic_virial=False)
    return r["energy"].reshape(()).clone()


dl = 1e-5
dEdl = ((energy_strained(dl) - energy_strained(-dl)) / (2 * dl)).item()
print("\n── virial (isotropic-strain) check ──")
print(f"  trace(model virial)   = {tr_model:.6e}")
print(f"  -dE/dλ (FD)           = {-dEdl:.6e}")
print(f"  |diff|                = {abs(tr_model - (-dEdl)):.3e}   rel={abs(tr_model-(-dEdl))/max(abs(dEdl),1e-6):.2e}")
print(f"  → virial {'PASS (matches -dE/dλ, Ξ_rec+Ξ_c captured)' if abs(tr_model-(-dEdl))/max(abs(dEdl),1e-6) < 1e-2 else 'FAIL'}")

