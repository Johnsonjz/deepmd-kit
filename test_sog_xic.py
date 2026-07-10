"""Strong-Ξ_c stress test: amplify the latent-charge head so charges depend strongly
on position (large ∂q/∂r → large Ξ_c), then confirm the decomposition virial
(F_full⊗r + box-strain) still matches the isotropic-strain finite difference (which
recomputes charges under strain → includes Ξ_c), and MEASURE Ξ_c by comparing to the
fixed-charge virial."""
import torch
torch.set_default_device("cpu")

from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.model.model.sog_model import SOGEnergyModel
from deepmd.pt.model.task.sog_energy_fitting import SOGEnergyFittingNet

dtype = torch.float64


def build(seed, amplify):
    torch.manual_seed(seed)
    desc = DescrptSeA(4.0, 3.5, [20, 20]).to(dtype)
    fit = SOGEnergyFittingNet(
        var_name="energy", ntypes=2, dim_descrpt=desc.get_dim_out(),
        dim_out_sr=1, dim_out_lr=1, mixed_types=desc.mixed_types(),
        n_dl=2, external_kspace=False,
    ).to(dtype)
    if amplify != 1.0:
        with torch.no_grad():
            for p in fit.parameters():
                p.mul_(amplify)
    model = SOGEnergyModel(descriptor=desc, fitting=fit, type_map=["A", "B"]).to(dtype)
    model.eval()
    return model


def energy(model, c, cell, atype):
    return model.forward(c, atype, box=cell, do_atomic_virial=False)["energy"].reshape(())


def check(model, coord, cell, atype, label):
    ret = model.forward(coord, atype, box=cell, do_atomic_virial=False)
    F = ret["force"][0].detach()
    vir = ret["virial"][0].reshape(3, 3).detach()
    tr_model = (vir[0, 0] + vir[1, 1] + vir[2, 2]).item()

    # force finite-difference
    dx = 1e-5; maxrel = 0.0
    for i in (0, 3):
        for d in range(3):
            cp = coord.clone(); cp[0, i, d] += dx
            cm = coord.clone(); cm[0, i, d] -= dx
            ffd = -((energy(model, cp, cell, atype) - energy(model, cm, cell, atype)) / (2 * dx)).item()
            maxrel = max(maxrel, abs(F[i, d].item() - ffd) / max(abs(ffd), 1e-6))

    # virial: trace vs -dE/dλ (isotropic strain recomputes charges → truth incl Ξ_c)
    dl = 1e-5
    def es(lam):
        return energy(model, coord * (1 + lam), cell * (1 + lam), atype)
    dEdl = ((es(dl) - es(-dl)) / (2 * dl)).item()
    vir_rel = abs(tr_model - (-dEdl)) / max(abs(dEdl), 1e-9)

    # Ξ_c magnitude: fixed-charge full-strain virial (charges held constant under strain).
    # Latent charges are set externally as fixed and the LR energy strained.
    q = ret["latent_charge"][0, :, 0].detach()
    import sog as sog_lib
    fit = model.atomic_model.fitting_net
    kern = sog_lib.Sog(sog_arguments={
        "use_atomwise": False, "amp": fit.amp.to(dtype), "bandwidth": fit.bandwidth.to(dtype),
        "kernel_param_mode": "internal", "kernel_tensor_mode": "external",
        "remove_self_interaction": bool(fit.remove_self_interaction),
        "nufft": False, "use_nufft": False, "use_cubes2_fft": False, "nlayers": 1,
        "norm_factor": 14.3996454784255, "trainable_kernel": False, "b": float(fit.b), "n_dl": 2.0,
    }, r_cut=4.0)
    nloc = coord.shape[1]
    batch = torch.zeros(nloc, dtype=torch.int64)
    strain = torch.zeros(1, 3, 3, dtype=dtype, requires_grad=True)
    sym = 0.5 * (strain + strain.transpose(-1, -2))
    pos_s = coord[0] + torch.einsum("ab,nb->na", sym[0], coord[0])
    box_s = cell[0] + torch.einsum("ab,kb->ka", sym[0], cell[0])
    Eb = kern(positions=pos_s, cell=box_s.unsqueeze(0), batch=batch,
              latent_charges=q.reshape(-1, 1), compute_energy=True, compute_bec=False)["E_lr"]
    dE = torch.autograd.grad(Eb.sum(), strain)[0]
    tr_fixedq = (-dE[0]).diagonal().sum().item()
    xi_c_frac = abs(tr_model - tr_fixedq) / max(abs(tr_model), 1e-9)

    print(f"  [{label}] |q|_rms={float((q**2).mean().sqrt()):.3f}  "
          f"force rel={maxrel:.2e}  virial rel(vs FD)={vir_rel:.2e}  "
          f"Ξ_c/Ξ_total≈{xi_c_frac:.1%}")
    return maxrel, vir_rel, xi_c_frac


torch.manual_seed(7)
coord = torch.rand((1, 6, 3), dtype=dtype) * 4.0
cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 6.0
atype = torch.tensor([[0, 0, 1, 1, 0, 1]], dtype=torch.int64)

print("decomposition virial (F_full⊗r + box-strain) vs isotropic-strain finite difference:")
ok = True
for amp in (1.0, 3.0, 6.0):
    m = build(seed=2031, amplify=amp)
    fr, vr, xc = check(m, coord, cell, atype, f"amplify×{amp}")
    ok = ok and fr < 1e-4 and vr < 1e-2
print(f"\n{'✓ decomposition captures Ξ_c across amplifications' if ok else '✗ FAIL'} "
      f"(force<1e-4, virial<1e-2 vs FD at all Ξ_c levels)")
