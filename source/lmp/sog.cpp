// SPDX-License-Identifier: LGPL-3.0-or-later

#include "sog.h"

#include <math.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

namespace {

constexpr double kSOGDefaultB = 1.62976708826776469;
constexpr double kSOGDefaultSigma = 2.180230445405648;
constexpr int kSOGDefaultM = 12;
constexpr double kSOGDefaultFinufftEps = 1e-9;

struct finufft_opts;
struct finufft_plan_s;
using finufft_plan = finufft_plan_s *;

struct FinufftApi {
  using makeplan_fn = int (*)(int, int, const int64_t *, int, int, double,
                              finufft_plan *, const finufft_opts *);
  using setpts_fn = int (*)(finufft_plan, int64_t, const double *, const double *,
                            const double *, int64_t, const double *,
                            const double *, const double *);
  using execute_fn = int (*)(finufft_plan, std::complex<double> *,
                             std::complex<double> *);
  using destroy_fn = int (*)(finufft_plan);

  void *handle = nullptr;
  makeplan_fn makeplan = nullptr;
  setpts_fn setpts = nullptr;
  execute_fn execute = nullptr;
  destroy_fn destroy = nullptr;
};

std::string to_lower_copy(const std::string &in) {
  std::string out = in;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

std::string finufft_build_hint() {
  return "build FINUFFT from https://github.com/flatironinstitute/finufft.git "
         "(for example: git clone ... && cd finufft && mkdir -p build && cd "
         "build && cmake .. -DBUILD_SHARED_LIBS=ON -DFINUFFT_STATIC_LINKING=OFF "
         "&& cmake --build . -j), then set kspace_style sog option "
         "finufft_library <path-to-libfinufft.so> or env "
         "DP_SOG_FINUFFT_LIBRARY";
}

int64_t mode_from_index(const int64_t idx, const int64_t nmode) {
  return idx - nmode / 2;
}

double to_periodic_angle(const double x, const double xlo, const double prd) {
  double frac = (x - xlo) / prd;
  frac -= std::floor(frac);
  double angle = MY_2PI * frac;
  if (angle >= MY_PI) {
    angle -= MY_2PI;
  }
  return angle;
}

#if !defined(_WIN32)
bool resolve_finufft_symbol(void *handle,
                            const char *name,
                            void **symbol,
                            std::string &error_msg) {
  dlerror();
  void *ptr = dlsym(handle, name);
  const char *err = dlerror();
  if (err != nullptr || ptr == nullptr) {
    error_msg = std::string("missing FINUFFT symbol ") + name;
    if (err != nullptr) {
      error_msg += std::string(": ") + err;
    }
    return false;
  }
  *symbol = ptr;
  return true;
}

bool try_load_finufft_api(const std::string &preferred,
                          FinufftApi &api,
                          std::string &error_msg) {
  std::string selected_lib;
  if (!preferred.empty()) {
    selected_lib = preferred;
  }

  const char *env_lib = std::getenv("DP_SOG_FINUFFT_LIBRARY");
  if (selected_lib.empty() && env_lib && std::string(env_lib).size() > 0) {
    selected_lib = env_lib;
  }

  if (selected_lib.empty()) {
    error_msg = "FINUFFT library path is not configured; " + finufft_build_hint();
    return false;
  }

  void *handle = dlopen(selected_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char *err = dlerror();
    error_msg = std::string("failed to load FINUFFT library ") + selected_lib;
    if (err != nullptr) {
      error_msg += std::string(": ") + err;
    }
    error_msg += "; " + finufft_build_hint();
    return false;
  }

  FinufftApi loaded;
  loaded.handle = handle;

  void *sym = nullptr;
  std::string local_error;

  if (!resolve_finufft_symbol(handle, "finufft_makeplan", &sym, local_error)) {
    dlclose(handle);
    error_msg = local_error;
    return false;
  }
  loaded.makeplan = reinterpret_cast<FinufftApi::makeplan_fn>(sym);

  if (!resolve_finufft_symbol(handle, "finufft_setpts", &sym, local_error)) {
    dlclose(handle);
    error_msg = local_error;
    return false;
  }
  loaded.setpts = reinterpret_cast<FinufftApi::setpts_fn>(sym);

  if (!resolve_finufft_symbol(handle, "finufft_execute", &sym, local_error)) {
    dlclose(handle);
    error_msg = local_error;
    return false;
  }
  loaded.execute = reinterpret_cast<FinufftApi::execute_fn>(sym);

  if (!resolve_finufft_symbol(handle, "finufft_destroy", &sym, local_error)) {
    dlclose(handle);
    error_msg = local_error;
    return false;
  }
  loaded.destroy = reinterpret_cast<FinufftApi::destroy_fn>(sym);

  api = loaded;
  return true;
}
#else
bool try_load_finufft_api(const std::string &, FinufftApi &, std::string &error_msg) {
  error_msg = "FINUFFT dynamic loading is only implemented on non-Windows builds";
  return false;
}
#endif

}  // namespace

SOGKSpace::SOGKSpace(LAMMPS *lmp)
    : PPPM(lmp),
      accuracy_in(1e-6),
      n_dl(1.0),
      remove_self_interaction(false),
      use_finufft(true),
      finufft_eps(kSOGDefaultFinufftEps),
      finufft_library(),
      finufft_warned(false),
      b_param(kSOGDefaultB),
      sigma_param(kSOGDefaultSigma),
      m_param(kSOGDefaultM),
      self_diag_sum(0.0),
      kernel_ready(false) {
  triclinic_support = 1;
}

SOGKSpace::~SOGKSpace() {}

bool SOGKSpace::is_keyword(const std::string &token) const {
  const std::string key = to_lower_copy(token);
  return key == "n_dl" || key == "remove_self_interaction" || key == "b" ||
         key == "sigma" || key == "m" || key == "amp" ||
         key == "bandwidth" || key == "use_finufft" ||
         key == "finufft_eps" || key == "finufft_library";
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

double SOGKSpace::kernel_prefactor(const double sqk) const {
  if (!(sqk > 0.0)) {
    return 0.0;
  }

  double coeff = 0.0;
  for (size_t mm = 0; mm < amp.size(); ++mm) {
    coeff += amp[mm] * std::exp(-0.5 * bandwidth[mm] * sqk);
  }

  return (4.0 * MY_PI * coeff / sqk);
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
  remove_self_interaction = false;
  use_finufft = true;
  finufft_eps = kSOGDefaultFinufftEps;
  finufft_library.clear();
  finufft_warned = false;
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
      use_finufft = val;
      iarg += 2;
    } else if (key == "finufft_eps") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing finufft_eps value");
      }
      finufft_eps = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "finufft_library") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing finufft_library value");
      }
      finufft_library = arg[iarg + 1];
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal kspace_style sog command");
    }
  }

  finalize_kernel_parameters();
}

void SOGKSpace::init() {
  if (differentiation_flag != 0) {
    error->all(FLERR,
               "kspace style sog currently supports only kspace_modify diff ik");
  }
  if (!kernel_ready) {
    finalize_kernel_parameters();
  }
  PPPM::init();
}

void SOGKSpace::setup() {
  PPPM::setup();
  if (domain->triclinic != 0) {
    // PPPM::setup_triclinic() calls non-virtual compute_gf_ik_triclinic().
    // Rebuild SOG Green's function here to replace Coulomb Green's function.
    rebuild_sog_greensfn();
  }
}

void SOGKSpace::compute_gf_ik() {
  // For orthorhombic cells PPPM::setup() dispatches here (virtual), so the
  // SOG kernel is wired directly into the PPPM Green's function build stage.
  rebuild_sog_greensfn();
}

void SOGKSpace::rebuild_sog_greensfn() {
  if (!kernel_ready) {
    finalize_kernel_parameters();
  }

  const double k_sq_max = (MY_2PI / n_dl) * (MY_2PI / n_dl);

  const double *const prd = domain->prd;
  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd * slab_volfactor;
  const double unitkx = (MY_2PI / xprd);
  const double unitky = (MY_2PI / yprd);
  const double unitkz = (MY_2PI / zprd_slab);
  const int domain_triclinic = domain->triclinic;

  int n = 0;

  if (domain_triclinic == 0) {
    for (int m = nzlo_fft; m <= nzhi_fft; ++m) {
      const int mper = m - nz_pppm * (2 * m / nz_pppm);

      for (int l = nylo_fft; l <= nyhi_fft; ++l) {
        const int lper = l - ny_pppm * (2 * l / ny_pppm);

        for (int k = nxlo_fft; k <= nxhi_fft; ++k) {
          const int kper = k - nx_pppm * (2 * k / nx_pppm);

          const double gkx = unitkx * kper;
          const double gky = unitky * lper;
          const double gkz = unitkz * mper;
          const double sqk = gkx * gkx + gky * gky + gkz * gkz;

          if (sqk > 0.0 && sqk <= k_sq_max) {
            const double kernel = kernel_prefactor(sqk);
            greensfn[n] = (std::isfinite(kernel) ? kernel : 0.0);
          } else {
            greensfn[n] = 0.0;
          }
          ++n;
        }
      }
    }
  } else {
    for (int m = nzlo_fft; m <= nzhi_fft; ++m) {
      const int mper = m - nz_pppm * (2 * m / nz_pppm);

      for (int l = nylo_fft; l <= nyhi_fft; ++l) {
        const int lper = l - ny_pppm * (2 * l / ny_pppm);

        for (int k = nxlo_fft; k <= nxhi_fft; ++k) {
          const int kper = k - nx_pppm * (2 * k / nx_pppm);

          double unitk_lamda[3];
          unitk_lamda[0] = MY_2PI * kper;
          unitk_lamda[1] = MY_2PI * lper;
          unitk_lamda[2] = MY_2PI * mper;
          x2lamdaT(&unitk_lamda[0], &unitk_lamda[0]);

          const double sqk = unitk_lamda[0] * unitk_lamda[0] +
                             unitk_lamda[1] * unitk_lamda[1] +
                             unitk_lamda[2] * unitk_lamda[2];

          if (sqk > 0.0 && sqk <= k_sq_max) {
            const double kernel = kernel_prefactor(sqk);
            greensfn[n] = (std::isfinite(kernel) ? kernel : 0.0);
          } else {
            greensfn[n] = 0.0;
          }
          ++n;
        }
      }
    }
  }

  self_diag_sum = 0.0;
}

bool SOGKSpace::compute_finufft(int eflag, int vflag) {
  if (!use_finufft) {
    return false;
  }

  const bool want_energy_global = (eflag & ENERGY_GLOBAL);
  const bool want_virial_global =
      (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR));

  if (domain->triclinic != 0 || comm->nprocs != 1) {
    if (!finufft_warned) {
      error->warning(
          FLERR,
          "kspace style sog FINUFFT path requires orthorhombic full-3D periodic "
          "single-rank run; this implementation does not yet support MPI-"
          "distributed FINUFFT solve. OpenMP threading inside one rank is still "
          "available via FINUFFT/OMP_NUM_THREADS. Falling back to PPPM path "
          "(domain->triclinic={}, comm->nprocs={})",
          domain->triclinic,
          comm->nprocs);
      finufft_warned = true;
    }
    return false;
  }

  const int nlocal = atom->nlocal;
  double qsqsum_local = 0.0;
  double *q = atom->q;
  for (int i = 0; i < nlocal; ++i) {
    qsqsum_local += q[i] * q[i];
  }

  if (qsqsum_local == 0.0) {
    energy = 0.0;
    for (int j = 0; j < 6; ++j) {
      virial[j] = 0.0;
    }
    return true;
  }

  struct FinufftLoaderState {
    bool attempted = false;
    bool available = false;
    FinufftApi api;
    std::string error;
  };
  static FinufftLoaderState loader;

  if (!loader.attempted) {
    loader.attempted = true;
    loader.available = try_load_finufft_api(finufft_library, loader.api, loader.error);
  }

  if (!loader.available) {
    if (!finufft_warned) {
      error->warning(
          FLERR,
          "kspace style sog failed to load FINUFFT ({}); falling back to PPPM "
          "path",
          loader.error);
      finufft_warned = true;
    }
    return false;
  }

  if (nlocal <= 0) {
    energy = 0.0;
    for (int j = 0; j < 6; ++j) {
      virial[j] = 0.0;
    }
    return true;
  }

  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  if (!(xprd > 0.0 && yprd > 0.0 && zprd > 0.0)) {
    if (!finufft_warned) {
      error->warning(FLERR,
                     "kspace style sog encountered non-positive box length in "
                     "FINUFFT path; falling back to PPPM path");
      finufft_warned = true;
    }
    return false;
  }

  const int nkx = std::max(1, static_cast<int>(xprd / n_dl));
  const int nky = std::max(1, static_cast<int>(yprd / n_dl));
  const int nkz = std::max(1, static_cast<int>(zprd / n_dl));

  const int64_t ms = static_cast<int64_t>(2 * nkx + 1);
  const int64_t mt = static_cast<int64_t>(2 * nky + 1);
  const int64_t mu = static_cast<int64_t>(2 * nkz + 1);
  const int64_t nmodes[3] = {ms, mt, mu};

  if (ms <= 0 || mt <= 0 || mu <= 0) {
    return false;
  }

  if (ms > std::numeric_limits<int64_t>::max() / mt ||
      ms * mt > std::numeric_limits<int64_t>::max() / mu) {
    if (!finufft_warned) {
      error->warning(FLERR,
                     "kspace style sog FINUFFT mode grid is too large; falling "
                     "back to PPPM path");
      finufft_warned = true;
    }
    return false;
  }
  const size_t ngrid = static_cast<size_t>(ms * mt * mu);

  std::vector<double> xj(static_cast<size_t>(nlocal));
  std::vector<double> yj(static_cast<size_t>(nlocal));
  std::vector<double> zj(static_cast<size_t>(nlocal));
  std::vector<std::complex<double>> q_complex(static_cast<size_t>(nlocal));

  double **x = atom->x;
  for (int i = 0; i < nlocal; ++i) {
    xj[static_cast<size_t>(i)] =
        to_periodic_angle(x[i][0], domain->boxlo[0], xprd);
    yj[static_cast<size_t>(i)] =
        to_periodic_angle(x[i][1], domain->boxlo[1], yprd);
    zj[static_cast<size_t>(i)] =
        to_periodic_angle(x[i][2], domain->boxlo[2], zprd);
    q_complex[static_cast<size_t>(i)] = std::complex<double>(q[i], 0.0);
  }

  std::vector<std::complex<double>> rho_k(ngrid, std::complex<double>(0.0, 0.0));

  finufft_plan plan1 = nullptr;
  int ier =
      loader.api.makeplan(1, 3, nmodes, -1, 1, finufft_eps, &plan1, nullptr);
  if (ier != 0 || plan1 == nullptr) {
    if (plan1 != nullptr) {
      loader.api.destroy(plan1);
    }
    if (!finufft_warned) {
      error->warning(
          FLERR,
          "kspace style sog FINUFFT makeplan(type1) failed with code {}; "
          "falling back to PPPM path",
          ier);
      finufft_warned = true;
    }
    return false;
  }

  ier = loader.api.setpts(plan1, static_cast<int64_t>(nlocal), xj.data(), yj.data(),
                          zj.data(), 0, nullptr, nullptr, nullptr);
  if (ier == 0) {
    ier = loader.api.execute(plan1, q_complex.data(), rho_k.data());
  }
  loader.api.destroy(plan1);
  if (ier != 0) {
    if (!finufft_warned) {
      error->warning(
          FLERR,
          "kspace style sog FINUFFT type1 execution failed with code {}; "
          "falling back to PPPM path",
          ier);
      finufft_warned = true;
    }
    return false;
  }

  const double k_sq_max = (MY_2PI / n_dl) * (MY_2PI / n_dl);
  const double twopi_over_x = MY_2PI / xprd;
  const double twopi_over_y = MY_2PI / yprd;
  const double twopi_over_z = MY_2PI / zprd;
  const double volume_local = xprd * yprd * zprd;

  double energy_local = 0.0;
  double diag_sum = 0.0;

  std::vector<std::complex<double>> grad_kx(
      ngrid, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> grad_ky(
      ngrid, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> grad_kz(
      ngrid, std::complex<double>(0.0, 0.0));

  for (int64_t iz = 0; iz < mu; ++iz) {
    const int64_t kz_mode = mode_from_index(iz, mu);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);
    for (int64_t iy = 0; iy < mt; ++iy) {
      const int64_t ky_mode = mode_from_index(iy, mt);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);
      for (int64_t ix = 0; ix < ms; ++ix) {
        const int64_t kx_mode = mode_from_index(ix, ms);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) {
          continue;
        }

        double kfac = 0.0;
        for (size_t mm = 0; mm < amp.size(); ++mm) {
          kfac += amp[mm] * std::exp(-0.5 * bandwidth[mm] * sqk);
        }
        if (!std::isfinite(kfac) || kfac == 0.0) {
          continue;
        }

        const size_t idx =
            static_cast<size_t>(ix + ms * (iy + mt * iz));
        const std::complex<double> rho = rho_k[idx];
        energy_local += kfac * std::norm(rho);
        diag_sum += kfac;

        const std::complex<double> conv = kfac * rho;
        grad_kx[idx] = std::complex<double>(0.0, kx) * conv;
        grad_ky[idx] = std::complex<double>(0.0, ky) * conv;
        grad_kz[idx] = std::complex<double>(0.0, kz) * conv;
      }
    }
  }

  energy_local /= (2.0 * volume_local);
  if (remove_self_interaction) {
    energy_local -= qsqsum_local * (diag_sum / (2.0 * volume_local));
  }

  finufft_plan plan2 = nullptr;
  ier = loader.api.makeplan(2, 3, nmodes, 1, 1, finufft_eps, &plan2, nullptr);
  if (ier != 0 || plan2 == nullptr) {
    if (plan2 != nullptr) {
      loader.api.destroy(plan2);
    }
    if (!finufft_warned) {
      error->warning(
          FLERR,
          "kspace style sog FINUFFT makeplan(type2) failed with code {}; "
          "falling back to PPPM path",
          ier);
      finufft_warned = true;
    }
    return false;
  }

  ier = loader.api.setpts(plan2, static_cast<int64_t>(nlocal), xj.data(), yj.data(),
                          zj.data(), 0, nullptr, nullptr, nullptr);

  std::vector<std::complex<double>> grad_x(static_cast<size_t>(nlocal));
  std::vector<std::complex<double>> grad_y(static_cast<size_t>(nlocal));
  std::vector<std::complex<double>> grad_z(static_cast<size_t>(nlocal));

  if (ier == 0) {
    ier = loader.api.execute(plan2, grad_x.data(), grad_kx.data());
  }
  if (ier == 0) {
    ier = loader.api.execute(plan2, grad_y.data(), grad_ky.data());
  }
  if (ier == 0) {
    ier = loader.api.execute(plan2, grad_z.data(), grad_kz.data());
  }
  loader.api.destroy(plan2);

  if (ier != 0) {
    if (!finufft_warned) {
      error->warning(
          FLERR,
          "kspace style sog FINUFFT type2 execution failed with code {}; "
          "falling back to PPPM path",
          ier);
      finufft_warned = true;
    }
    return false;
  }

  const double qscale_local = force->qqrd2e;
  double **f = atom->f;
  std::array<double, 6> virial_acc = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < nlocal; ++i) {
    const double qi = q[i];
    const double fx = -qi * grad_x[static_cast<size_t>(i)].real() / volume_local;
    const double fy = -qi * grad_y[static_cast<size_t>(i)].real() / volume_local;
    const double fz = -qi * grad_z[static_cast<size_t>(i)].real() / volume_local;

    const double fxs = qscale_local * fx;
    const double fys = qscale_local * fy;
    const double fzs = qscale_local * fz;

    f[i][0] += fxs;
    f[i][1] += fys;
    f[i][2] += fzs;

    if (want_virial_global) {
      const double rx = x[i][0];
      const double ry = x[i][1];
      const double rz = x[i][2];
      virial_acc[0] += rx * fxs;
      virial_acc[1] += ry * fys;
      virial_acc[2] += rz * fzs;
      virial_acc[3] += rx * fys;
      virial_acc[4] += rx * fzs;
      virial_acc[5] += ry * fzs;
    }
  }

  if (want_energy_global) {
    energy = qscale_local * energy_local;
  } else {
    energy = 0.0;
  }

  if (want_virial_global) {
    for (int j = 0; j < 6; ++j) {
      virial[j] = virial_acc[static_cast<size_t>(j)];
    }
  } else {
    for (int j = 0; j < 6; ++j) {
      virial[j] = 0.0;
    }
  }

  return true;
}

void SOGKSpace::compute(int eflag, int vflag) {
  if (compute_finufft(eflag, vflag)) {
    return;
  }
  PPPM::compute(eflag, vflag);
}

void SOGKSpace::fieldforce_ik() {
  PPPM::fieldforce_ik();
}

double SOGKSpace::memory_usage() {
  return PPPM::memory_usage() +
         static_cast<double>((amp.size() + bandwidth.size()) * sizeof(double));
}
