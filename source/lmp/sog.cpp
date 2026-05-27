// SPDX-License-Identifier: LGPL-3.0-or-later

#include "sog.h"

#include <math.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "math_const.h"
#include "pair.h"

using namespace LAMMPS_NS;
using namespace MathConst;

namespace {

constexpr double kSOGDefaultB = 1.62976708826776469;
constexpr double kSOGDefaultSigma = 2.180230445405648;
constexpr int kSOGDefaultM = 12;
constexpr double kSOGDefaultFinufftEps = 1e-9;
constexpr int kSOGMeshAliasExtent = 8;
constexpr int kPPPMGridOffset = 16384;

std::string to_lower_copy(const std::string &in) {
  std::string out = in;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

double sinc(const double x) {
  if (std::fabs(x) < 1e-14) {
    return 1.0;
  }
  return std::sin(x) / x;
}

double sinc_pow(const double x, const int p) {
  const double s = sinc(x);
  return std::pow(s, static_cast<double>(p));
}

}  // namespace

SOGKSpace::SOGKSpace(LAMMPS *lmp)
  : PPPM(lmp),
      accuracy_in(1e-6),
      n_dl(1.0),
      n_dl_user_specified(false),
      n_dl_from_model(false),
      remove_self_interaction(false),
      use_finufft(false),
      finufft_eps(kSOGDefaultFinufftEps),
      finufft_library(),
      finufft_warned(false),
      mesh_oversample(1.5),
      mesh_alias_extent(kSOGMeshAliasExtent),
      b_param(kSOGDefaultB),
      sigma_param(kSOGDefaultSigma),
      m_param(kSOGDefaultM),
      self_diag_sum(0.0),
      kernel_ready(false) {
  triclinic_support = 0;
}

SOGKSpace::~SOGKSpace() = default;

bool SOGKSpace::is_keyword(const std::string &token) const {
  const std::string key = to_lower_copy(token);
  return key == "n_dl" || key == "n_dl_from_model" ||
         key == "remove_self_interaction" || key == "b" || key == "sigma" ||
         key == "m" || key == "amp" || key == "bandwidth" ||
         key == "use_finufft" || key == "finufft_eps" ||
         key == "finufft_library" || key == "mesh_oversample" ||
         key == "mesh_alias_extent";
}

bool SOGKSpace::parse_bool_token(const std::string &token, bool &value) const {
  const std::string opt = to_lower_copy(token);
  if (opt == "1" || opt == "yes" || opt == "on" || opt == "true") {
    value = true;
    return true;
  }
  if (opt == "0" || opt == "no" || opt == "off" || opt == "false") {
    value = false;
    return true;
  }
  return false;
}

bool SOGKSpace::try_import_n_dl_from_pair_model(const bool strict_missing) {
  if (force == nullptr || force->pair == nullptr) {
    if (strict_missing) {
      error->all(FLERR,
                 "kspace style sog n_dl_from_model requires pair_style deepmd");
    }
    return false;
  }

  int dim = 0;
  void *const raw = force->pair->extract("deepmd_model_n_dl", dim);
  if (raw == nullptr) {
    if (strict_missing) {
      error->all(
          FLERR,
          "kspace style sog n_dl_from_model could not find n_dl in model metadata");
    }
    return false;
  }

  const double model_n_dl = *static_cast<double *>(raw);
  if (!(std::isfinite(model_n_dl) && model_n_dl > 0.0)) {
    if (strict_missing) {
      error->all(
          FLERR,
          "kspace style sog n_dl_from_model found invalid model n_dl value");
    }
    return false;
  }

  n_dl = model_n_dl;
  return true;
}

void SOGKSpace::finalize_kernel_parameters() {
  if (m_param < 1) {
    error->all(FLERR, "kspace style sog requires M >= 1");
  }
  if (!(std::isfinite(n_dl) && n_dl > 0.0)) {
    error->all(FLERR, "kspace style sog requires n_dl > 0");
  }
  if (!(std::isfinite(b_param) && b_param > 0.0)) {
    error->all(FLERR, "kspace style sog requires b > 0");
  }
  if (!(std::isfinite(sigma_param) && sigma_param > 0.0)) {
    error->all(FLERR, "kspace style sog requires sigma > 0");
  }
  if (!(std::isfinite(finufft_eps) && finufft_eps > 0.0)) {
    error->all(FLERR, "kspace style sog requires finufft_eps > 0");
  }
  if (!(std::isfinite(mesh_oversample) && mesh_oversample >= 1.0)) {
    error->all(FLERR, "kspace style sog requires mesh_oversample >= 1");
  }
  if (mesh_alias_extent < 1) {
    error->all(FLERR, "kspace style sog requires mesh_alias_extent >= 1");
  }

  if (bandwidth.empty()) {
    bandwidth.resize(static_cast<size_t>(m_param), 0.0);
    for (int mm = 0; mm < m_param; ++mm) {
      const double bw = sigma_param * std::pow(b_param, static_cast<double>(mm));
      bandwidth[static_cast<size_t>(mm)] = bw * bw;
    }
  }

  if (amp.empty()) {
    const double amp0 = 4.0 * MY_PI * std::log(b_param);
    amp.assign(bandwidth.size(), amp0);
  }

  if (amp.size() == 1 && bandwidth.size() > 1) {
    amp.assign(bandwidth.size(), amp[0]);
  }

  if (amp.size() != bandwidth.size()) {
    error->all(
        FLERR,
        "kspace style sog requires amp to be scalar or same length as bandwidth");
  }

  for (size_t ii = 0; ii < amp.size(); ++ii) {
    if (!std::isfinite(amp[ii])) {
      error->all(FLERR, "kspace style sog found non-finite amp value");
    }
  }
  for (size_t ii = 0; ii < bandwidth.size(); ++ii) {
    if (!(std::isfinite(bandwidth[ii]) && bandwidth[ii] > 0.0)) {
      error->all(
          FLERR,
          "kspace style sog requires all bandwidth values to be positive finite");
    }
  }

  kernel_ready = true;
}

double SOGKSpace::spectral_kernel(const double sqk) const {
  if (!(sqk > 0.0)) {
    return 0.0;
  }

  double coeff = 0.0;
  for (size_t mm = 0; mm < amp.size(); ++mm) {
    coeff += amp[mm] * std::exp(-0.5 * bandwidth[mm] * sqk);
  }

  return coeff;
}

void SOGKSpace::settings(int narg, char **arg) {
  if (narg < 1) {
    error->all(FLERR, "Illegal kspace_style sog command");
  }

  accuracy_in = std::fabs(atof(arg[0]));
  if (!(std::isfinite(accuracy_in) && accuracy_in > 0.0)) {
    error->all(FLERR, "kspace style sog requires a positive accuracy argument");
  }

  char *base_arg[1];
  base_arg[0] = arg[0];
  PPPM::settings(1, base_arg);

  n_dl = 1.0;
  n_dl_user_specified = false;
  n_dl_from_model = false;
  remove_self_interaction = false;
  use_finufft = false;
  finufft_eps = kSOGDefaultFinufftEps;
  finufft_library.clear();
  finufft_warned = false;
  mesh_oversample = 1.5;
  mesh_alias_extent = kSOGMeshAliasExtent;
  b_param = kSOGDefaultB;
  sigma_param = kSOGDefaultSigma;
  m_param = kSOGDefaultM;
  amp.clear();
  bandwidth.clear();
  kernel_ready = false;

  int iarg = 1;
  while (iarg < narg) {
    const std::string key = to_lower_copy(arg[iarg]);
    if (key == "n_dl") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing n_dl value");
      }
      n_dl = atof(arg[iarg + 1]);
      n_dl_user_specified = true;
      iarg += 2;
    } else if (key == "n_dl_from_model") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing n_dl_from_model value");
      }
      bool val = false;
      if (!parse_bool_token(arg[iarg + 1], val)) {
        error->all(FLERR, "kspace style sog n_dl_from_model expects yes/no token");
      }
      n_dl_from_model = val;
      iarg += 2;
    } else if (key == "remove_self_interaction") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing remove_self_interaction value");
      }
      bool val = false;
      if (!parse_bool_token(arg[iarg + 1], val)) {
        error->all(
            FLERR,
            "kspace style sog remove_self_interaction expects yes/no token");
      }
      remove_self_interaction = val;
      iarg += 2;
    } else if (key == "b") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing b value");
      }
      b_param = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "sigma") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing sigma value");
      }
      sigma_param = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "m") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing M value");
      }
      m_param = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "amp") {
      amp.clear();
      ++iarg;
      while (iarg < narg && !is_keyword(arg[iarg])) {
        amp.push_back(atof(arg[iarg]));
        ++iarg;
      }
      if (amp.empty()) {
        error->all(FLERR, "kspace style sog amp expects at least one value");
      }
    } else if (key == "bandwidth") {
      bandwidth.clear();
      ++iarg;
      while (iarg < narg && !is_keyword(arg[iarg])) {
        bandwidth.push_back(atof(arg[iarg]));
        ++iarg;
      }
      if (bandwidth.empty()) {
        error->all(FLERR, "kspace style sog bandwidth expects at least one value");
      }
    } else if (key == "use_finufft") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing use_finufft value");
      }
      bool val = false;
      if (!parse_bool_token(arg[iarg + 1], val)) {
        error->all(FLERR, "kspace style sog use_finufft expects yes/no token");
      }
      use_finufft = false;
      if (val && comm->me == 0) {
        error->warning(
            FLERR,
            "kspace style sog PPPM path ignores use_finufft=yes; keeping "
            "mesh-FFT PPPM solver");
      }
      iarg += 2;
    } else if (key == "finufft_eps") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing finufft_eps value");
      }
      finufft_eps = atof(arg[iarg + 1]);
      if (comm->me == 0) {
        error->warning(
            FLERR,
            "kspace style sog PPPM path ignores finufft_eps option "
            "(compatibility parse only)");
      }
      iarg += 2;
    } else if (key == "finufft_library") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing finufft_library value");
      }
      finufft_library = arg[iarg + 1];
      if (comm->me == 0) {
        error->warning(
            FLERR,
            "kspace style sog PPPM path ignores finufft_library option "
            "(compatibility parse only)");
      }
      iarg += 2;
    } else if (key == "mesh_oversample") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing mesh_oversample value");
      }
      mesh_oversample = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "mesh_alias_extent") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing mesh_alias_extent value");
      }
      mesh_alias_extent = atoi(arg[iarg + 1]);
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal kspace_style sog command");
    }
  }

  finalize_kernel_parameters();
}

void SOGKSpace::init() {
  if (!n_dl_user_specified) {
    const bool strict = n_dl_from_model;
    if (try_import_n_dl_from_pair_model(strict)) {
      kernel_ready = false;
    }
  } else if (n_dl_from_model && comm->me == 0) {
    error->warning(
        FLERR,
        "kspace style sog ignores n_dl_from_model because explicit n_dl is set");
  }

  if (!kernel_ready) {
    finalize_kernel_parameters();
  }

  if (differentiation_flag != 0) {
    error->all(FLERR,
               "kspace style sog currently supports only kspace_modify diff ik");
  }
  if (domain->triclinic != 0) {
    error->all(FLERR,
               "kspace style sog currently supports only orthorhombic boxes");
  }
  if (slabflag != 0) {
    error->all(FLERR,
               "kspace style sog currently requires fully periodic boundaries");
  }

  if (use_finufft) {
    if (comm->me == 0) {
      error->warning(
          FLERR,
          "kspace style sog PPPM-level skeleton currently ignores use_finufft and "
          "uses PPPM mesh path only");
    }
    use_finufft = false;
  }

  // Keep g_ewald finite and fixed. SOG uses custom Green functions and does not
  // rely on PPPM's Ewald tuning formulas.
  g_ewald = 1.0;
  gewaldflag = 1;

  PPPM::init();
}

void SOGKSpace::setup() {
  PPPM::setup();
}

void SOGKSpace::set_grid_global() {
  if (!(std::isfinite(n_dl) && n_dl > 0.0)) {
    error->all(FLERR, "kspace style sog requires n_dl > 0");
  }
  if (!(std::isfinite(mesh_oversample) && mesh_oversample >= 1.0)) {
    error->all(FLERR, "kspace style sog requires mesh_oversample >= 1");
  }
  if (domain->triclinic != 0) {
    error->all(FLERR,
               "kspace style sog currently supports only orthorhombic boxes");
  }

  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double zprd_slab = zprd * slab_volfactor;

  const double mesh_scale = std::max(1.0, mesh_oversample);
  nx_pppm = std::max(2,
                     static_cast<int>(std::ceil(mesh_scale * 2.0 * xprd / n_dl)));
  ny_pppm = std::max(2,
                     static_cast<int>(std::ceil(mesh_scale * 2.0 * yprd / n_dl)));
  nz_pppm = std::max(
      2, static_cast<int>(std::ceil(mesh_scale * 2.0 * zprd_slab / n_dl)));

  if (nx_pppm & 1) {
    ++nx_pppm;
  }
  if (ny_pppm & 1) {
    ++ny_pppm;
  }
  if (nz_pppm & 1) {
    ++nz_pppm;
  }

  while (!factorable(nx_pppm)) {
    ++nx_pppm;
  }
  while (!factorable(ny_pppm)) {
    ++ny_pppm;
  }
  while (!factorable(nz_pppm)) {
    ++nz_pppm;
  }

  h_x = xprd / nx_pppm;
  h_y = yprd / ny_pppm;
  h_z = zprd_slab / nz_pppm;

  if (nx_pppm >= kPPPMGridOffset || ny_pppm >= kPPPMGridOffset ||
      nz_pppm >= kPPPMGridOffset) {
    error->all(FLERR, "PPPM grid is too large");
  }
}

void SOGKSpace::compute_gf_ik() {
  const double *const prd = domain->prd;
  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd * slab_volfactor;

  const double unitkx = (MY_2PI / xprd);
  const double unitky = (MY_2PI / yprd);
  const double unitkz = (MY_2PI / zprd_slab);
  const double k_sq_max = (MY_2PI / n_dl) * (MY_2PI / n_dl);
  const int twoorder = 2 * order;
  const int alias_extent = mesh_alias_extent;

  if (greensfn_energy.size() != static_cast<size_t>(nfft)) {
    greensfn_energy.assign(static_cast<size_t>(nfft), 0.0);
  }

  int n = 0;
  double diag_sum_local = 0.0;

  for (int m = nzlo_fft; m <= nzhi_fft; ++m) {
    const int mper = m - nz_pppm * (2 * m / nz_pppm);
    const double kz = unitkz * static_cast<double>(mper);
    const double snz = std::pow(std::sin(0.5 * kz * h_z), 2.0);

    for (int l = nylo_fft; l <= nyhi_fft; ++l) {
      const int lper = l - ny_pppm * (2 * l / ny_pppm);
      const double ky = unitky * static_cast<double>(lper);
      const double sny = std::pow(std::sin(0.5 * ky * h_y), 2.0);

      for (int k = nxlo_fft; k <= nxhi_fft; ++k) {
        const int kper = k - nx_pppm * (2 * k / nx_pppm);
        const double kx = unitkx * static_cast<double>(kper);
        const double snx = std::pow(std::sin(0.5 * kx * h_x), 2.0);

        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) {
          greensfn[n] = 0.0;
          greensfn_energy[static_cast<size_t>(n)] = 0.0;
          ++n;
          continue;
        }

        const double denominator = gf_denom(snx, sny, snz);
        if (!(denominator > 1e-24) || !std::isfinite(denominator)) {
          greensfn[n] = 0.0;
          greensfn_energy[static_cast<size_t>(n)] = 0.0;
          ++n;
          continue;
        }

        double sum0 = 0.0;
        double sum1 = 0.0;
        for (int nx_alias = -alias_extent; nx_alias <= alias_extent; ++nx_alias) {
          const double qx =
              unitkx * static_cast<double>(kper + nx_pppm * nx_alias);
          const double wx = sinc_pow(0.5 * qx * h_x, twoorder);

          for (int ny_alias = -alias_extent; ny_alias <= alias_extent;
               ++ny_alias) {
            const double qy =
                unitky * static_cast<double>(lper + ny_pppm * ny_alias);
            const double wy = sinc_pow(0.5 * qy * h_y, twoorder);

            for (int nz_alias = -alias_extent; nz_alias <= alias_extent;
                 ++nz_alias) {
              const double qz =
                  unitkz * static_cast<double>(mper + nz_pppm * nz_alias);
              const double wz = sinc_pow(0.5 * qz * h_z, twoorder);

              const double qsq = qx * qx + qy * qy + qz * qz;
              if (!(qsq > 0.0 && qsq <= k_sq_max)) {
                continue;
              }

              const double kfac = spectral_kernel(qsq);
              if (!std::isfinite(kfac) || kfac == 0.0) {
                continue;
              }

              const double w2 = wx * wy * wz;
              sum0 += kfac * w2;
              const double dot1 = kx * qx + ky * qy + kz * qz;
              sum1 += dot1 * kfac * w2;
            }
          }
        }

        const double geff_energy = sum0 / denominator;
        const double geff_force = sum1 / (sqk * denominator);
        if (!std::isfinite(geff_energy) || !std::isfinite(geff_force)) {
          greensfn[n] = 0.0;
          greensfn_energy[static_cast<size_t>(n)] = 0.0;
          ++n;
          continue;
        }

        const double w2_principal =
            sinc_pow(0.5 * kx * h_x, twoorder) *
            sinc_pow(0.5 * ky * h_y, twoorder) *
            sinc_pow(0.5 * kz * h_z, twoorder);
        if (w2_principal > 1e-20) {
          diag_sum_local += sum0 / w2_principal;
        }

        greensfn[n] = geff_force;
        greensfn_energy[static_cast<size_t>(n)] = geff_energy;
        ++n;
      }
    }
  }

  double diag_sum_all = 0.0;
  MPI_Allreduce(&diag_sum_local, &diag_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);
  self_diag_sum = diag_sum_all / (2.0 * volume);
}

void SOGKSpace::poisson_ik() {
  int i, j, k, n;
  double eng;

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = 0.0;
  }

  fft1->compute(work1, work1, FFT3d::FORWARD);

  const bigint ngridtotal = static_cast<bigint>(nx_pppm) * ny_pppm * nz_pppm;
  const double scaleinv = 1.0 / static_cast<double>(ngridtotal);
  const double s2 = scaleinv * scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft; i++) {
        eng = s2 * greensfn_energy[static_cast<size_t>(i)] *
              (work1[n] * work1[n] + work1[n + 1] * work1[n + 1]);
        for (j = 0; j < 6; j++) {
          virial[j] += eng * vg[i][j];
        }
        if (eflag_global) {
          energy += eng;
        }
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft; i++) {
        energy += s2 * greensfn_energy[static_cast<size_t>(i)] *
                  (work1[n] * work1[n] + work1[n + 1] * work1[n + 1]);
        n += 2;
      }
    }
  }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  if (evflag_atom) {
    poisson_peratom();
  }

  if (triclinic) {
    poisson_ik_triclinic();
    return;
  }

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fkx[i] * work1[n + 1];
        work2[n + 1] = fkx[i] * work1[n];
        n += 2;
      }

  fft2->compute(work2, work2, FFT3d::BACKWARD);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fky[j] * work1[n + 1];
        work2[n + 1] = fky[j] * work1[n];
        n += 2;
      }

  fft2->compute(work2, work2, FFT3d::BACKWARD);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fkz[k] * work1[n + 1];
        work2[n + 1] = fkz[k] * work1[n];
        n += 2;
      }

  fft2->compute(work2, work2, FFT3d::BACKWARD);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

void SOGKSpace::compute(int eflag, int vflag) {
  const bool want_virial_global = (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR));
  const int nlocal = atom->nlocal;
  std::vector<double> force_before;
  if (want_virial_global && nlocal > 0) {
    force_before.resize(static_cast<size_t>(3 * nlocal), 0.0);
    for (int i = 0; i < nlocal; ++i) {
      force_before[static_cast<size_t>(3 * i + 0)] = atom->f[i][0];
      force_before[static_cast<size_t>(3 * i + 1)] = atom->f[i][1];
      force_before[static_cast<size_t>(3 * i + 2)] = atom->f[i][2];
    }
  }

  // Disable PPPM's built-in global virial accumulation (Ewald-form vg).
  // We reconstruct SOG-consistent global virial from this kspace force increment.
  const int vflag_pppm = vflag & ~(VIRIAL_PAIR | VIRIAL_FDOTR);
  PPPM::compute(eflag, vflag_pppm);

  if (!eflag_global || qsqsum == 0.0) {
    // Still rebuild virial when requested even if energy flag is off.
    if (!want_virial_global) {
      return;
    }
  }

  if (eflag_global && qsqsum != 0.0) {
    const double qscale = qqrd2e * scale;
    // PPPM::compute always applies Ewald self/background terms. Undo them for SOG.
    energy += qscale *
              (g_ewald * qsqsum / MY_PIS +
               MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume));

    if (remove_self_interaction) {
      energy -= qscale * qsqsum * self_diag_sum;
    }
  }

  if (want_virial_global) {
    std::array<double, 6> virial_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double **x = atom->x;
    double **f = atom->f;
    for (int i = 0; i < nlocal; ++i) {
      const double dfx =
          f[i][0] - force_before[static_cast<size_t>(3 * i + 0)];
      const double dfy =
          f[i][1] - force_before[static_cast<size_t>(3 * i + 1)];
      const double dfz =
          f[i][2] - force_before[static_cast<size_t>(3 * i + 2)];

      virial_local[0] += x[i][0] * dfx;
      virial_local[1] += x[i][1] * dfy;
      virial_local[2] += x[i][2] * dfz;
      virial_local[3] += x[i][0] * dfy;
      virial_local[4] += x[i][0] * dfz;
      virial_local[5] += x[i][1] * dfz;
    }

    double virial_all[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(virial_local.data(), virial_all, 6, MPI_DOUBLE, MPI_SUM, world);
    for (int j = 0; j < 6; ++j) {
      virial[j] = virial_all[j];
    }
  }
}

double SOGKSpace::memory_usage() {
  const double pppm_bytes = PPPM::memory_usage();
  const size_t param_bytes =
      (amp.capacity() + bandwidth.capacity() + greensfn_energy.capacity()) *
      sizeof(double);
  return pppm_bytes + static_cast<double>(param_bytes);
}
