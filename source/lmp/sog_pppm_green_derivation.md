# SOG Discrete Green Function on Internal Mesh-FFT Path (Rigorous Form)

For build, input-script, validation, and debug workflow, see
`source/lmp/sog_les_build_run_debug_workflow.md`.

This note gives a strict derivation for the internal SOG mesh-FFT path in
`source/lmp/sog.cpp`.

## 1. Problem setup and notation

Assume a 3D periodic orthorhombic box

$$
\Omega = [0,L_x)\times[0,L_y)\times[0,L_z),\qquad V=L_xL_yL_z.
$$

Point charges are

$$
\rho(\mathbf r)=\sum_{i=1}^{N_a} q_i\,\delta(\mathbf r-\mathbf r_i).
$$

Continuous Fourier coefficients (periodic convention) are

$$
\hat\rho(\mathbf q)=\int_\Omega \rho(\mathbf r)e^{-i\mathbf q\cdot\mathbf r}d\mathbf r
=\sum_i q_i e^{-i\mathbf q\cdot\mathbf r_i},
$$

for reciprocal vectors

$$
\mathbf q(\mathbf n)=2\pi\left(\frac{n_x}{L_x},\frac{n_y}{L_y},\frac{n_z}{L_z}\right),\quad
\mathbf n\in\mathbb Z^3.
$$

SOG spectral kernel:

$$
\Phi(s)=\sum_{m=1}^{M} a_m\exp\left(-\frac{1}{2}b_m s\right),\qquad s\ge 0,
$$

with reciprocal cutoff

$$
|\mathbf q|\le k_{\max},\qquad k_{\max}=\frac{2\pi}{n_{dl}}.
$$

## 2. Continuous reference energy and force

Reference reciprocal energy:

$$
E_{\mathrm{ref}}=
\frac{1}{2V}\sum_{\substack{\mathbf q\neq 0\\|\mathbf q|\le k_{\max}}}
\Phi(|\mathbf q|^2)\,|\hat\rho(\mathbf q)|^2.
$$

Reference force on atom $i$:

$$
\mathbf F_i^{\mathrm{ref}}=
-\frac{q_i}{V}
\sum_{\substack{\mathbf q\neq 0\\|\mathbf q|\le k_{\max}}}
i\mathbf q\,\Phi(|\mathbf q|^2)\,\hat\rho(\mathbf q)e^{i\mathbf q\cdot\mathbf r_i}.
$$

The FINUFFT path evaluates this model directly.

## 3. Mesh, assignment, and FFT normalization

Let mesh sizes be $(n_x,n_y,n_z)$, spacings

$$
d_x=L_x/n_x,\quad d_y=L_y/n_y,\quad d_z=L_z/n_z,
$$

and total mesh count $N=n_xn_yn_z$, cell volume $dV=V/N$.

Nyquist condition for representing $k_{\max}$ is

$$
\frac{\pi}{d_\alpha}\ge k_{\max}
\iff d_\alpha\le \frac{n_{dl}}{2},\qquad \alpha\in\{x,y,z\}.
$$

In implementation, a controllable oversampling factor $s_{mesh}\ge 1$ is used:

$$
n_\alpha\approx \left\lceil 2s_{mesh}\frac{L_\alpha}{n_{dl}}\right\rceil
	ext{ (rounded to even)}.
$$

Order-$p$ cardinal B-spline assignment (current $p=5$) deposits charge density:

$$
\rho_g=\frac{1}{dV}\sum_i q_i W_p(\mathbf r_g-\mathbf r_i).
$$

Unnormalized DFT pair used by `FFT3d`:

$$
\hat\rho_{\mathbf k}=\sum_{\mathbf g}\rho_{\mathbf g}
e^{-2\pi i(k_x g_x/n_x+k_y g_y/n_y+k_z g_z/n_z)},
$$

$$
\rho_{\mathbf g}=\frac{1}{N}\sum_{\mathbf k}\hat\rho_{\mathbf k}
e^{2\pi i(k_x g_x/n_x+k_y g_y/n_y+k_z g_z/n_z)}.
$$

## 4. Alias model induced by assignment

For principal mode $\mathbf q_{\mathbf k}$ and alias index $\mathbf u\in\mathbb Z^3$, define

$$
\mathbf K_{\mathbf u}=2\pi\left(\frac{u_x}{d_x},\frac{u_y}{d_y},\frac{u_z}{d_z}\right),
\qquad
\mathbf q_{\mathbf k,\mathbf u}=\mathbf q_{\mathbf k}+\mathbf K_{\mathbf u}.
$$

Assignment transfer function and power response:

$$
U(\mathbf q)=\prod_{\alpha\in\{x,y,z\}}
\operatorname{sinc}\left(\frac{q_\alpha d_\alpha}{2}\right)^p,
\qquad
W_2(\mathbf q)=|U(\mathbf q)|^2.
$$

Then mesh Fourier mode admits the alias decomposition

$$
\hat\rho_{\mathbf k}=\sum_{\mathbf u\in\mathbb Z^3}
U(\mathbf q_{\mathbf k,\mathbf u})\,\hat\rho(\mathbf q_{\mathbf k,\mathbf u}).
$$

Define

$$
S(\mathbf k)=\sum_{\mathbf u}W_2(\mathbf q_{\mathbf k,\mathbf u}).
$$

## 5. Discrete Green functions for force and energy

With $ik$ differentiation, write reciprocal gradient field as

$$
\widehat{\nabla\phi}(\mathbf k)=i\mathbf q_{\mathbf k}
\,G_F(\mathbf k)\,\frac{\hat\rho_{\mathbf k}}{N}.
$$

Using the standard Hockney-Eastwood alias-error projection for force,

$$
G_F(\mathbf k)=
\frac{N_F(\mathbf k)}{|\mathbf q_{\mathbf k}|^2 S(\mathbf k)^2},
$$

$$
N_F(\mathbf k)=\sum_{\mathbf u}
\big(\mathbf q_{\mathbf k}\cdot\mathbf q_{\mathbf k,\mathbf u}\big)
\Phi\!\left(|\mathbf q_{\mathbf k,\mathbf u}|^2\right)
W_2(\mathbf q_{\mathbf k,\mathbf u}),
$$

with terms restricted to $|\mathbf q_{\mathbf k,\mathbf u}|\le k_{\max}$.

For energy, use the consistent quadratic form

$$
G_E(\mathbf k)=\frac{N_E(\mathbf k)}{S(\mathbf k)^2},
$$

$$
N_E(\mathbf k)=\sum_{\mathbf u}
\Phi\!\left(|\mathbf q_{\mathbf k,\mathbf u}|^2\right)
W_2(\mathbf q_{\mathbf k,\mathbf u}),
$$

and accumulate

$$
E_{\mathrm{mesh}}=
\frac{V}{2}
\sum_{\mathbf k\neq 0}
\frac{|\hat\rho_{\mathbf k}|^2}{N^2}G_E(\mathbf k).
$$

In the no-alias limit ($\mathbf u=0$ only), both kernels reduce to
$\Phi(|\mathbf q_{\mathbf k}|^2)$.

## 6. Gathered forces in real space

After inverse FFT of $\widehat{\nabla\phi}(\mathbf k)$, interpolate with the same
assignment weights:

$$
\nabla\phi(\mathbf r_i)\approx\sum_{\mathbf g}W_{ig}\,\nabla\phi_{\mathbf g}.
$$

Final force (LAMMPS unit conversion included):

$$
\mathbf F_i=-q_i\,\texttt{qqrd2e}\,\nabla\phi(\mathbf r_i).
$$

## 7. Self-interaction subtraction (RSI)

For `remove_self_interaction yes`, energy uses

$$
E=E_{\mathrm{mesh}}-\left(\sum_i q_i^2\right)\,\texttt{self\_diag\_sum},
$$

with

$$
	exttt{self\_diag\_sum}=
\frac{1}{2V}\sum_{\mathbf k\neq 0}D_{\mathrm{self}}(\mathbf k),
$$

$$
D_{\mathrm{self}}(\mathbf k)=
\frac{N_E(\mathbf k)}{W_2(\mathbf q_{\mathbf k})}.
$$

Again, in the no-alias limit, $D_{\mathrm{self}}(\mathbf k)\to
\Phi(|\mathbf q_{\mathbf k}|^2)$.

## 8. Truncated alias sum and controllable accuracy

Implementation uses symmetric truncation

$$
u_x,u_y,u_z\in[-U_{\max},U_{\max}].
$$

The neglected tail is exponentially damped by $\Phi$ and additionally suppressed
by B-spline factors $W_2$:

$$
	ext{tail} \lesssim
\sum_{\|\mathbf u\|_\infty>U_{\max}}
\exp\big(-c_1\|\mathbf u\|^2\big)
\,(1+\|\mathbf u\|)^{-2p},
$$

so increasing either $s_{mesh}$ or $U_{\max}$ improves accuracy.

Current code exposes:

- `mesh_oversample` (default `1.5`)
- `mesh_alias_extent` (default `8`)

## 9. Code mapping (for audit)

- $G_F$: `geff` in `compute_mesh_fft`
- $G_E$: `geff_energy` in `compute_mesh_fft`
- $D_{self}$ accumulation: `diag_sum_local`
- $s_{mesh}$: `mesh_oversample`
- $U_{\max}$: `mesh_alias_extent`

## 10. Implementation limits

- Mesh path supports orthorhombic boxes only.
- Mesh path supports single MPI rank only.
- FINUFFT remains the high-accuracy reference implementation.
