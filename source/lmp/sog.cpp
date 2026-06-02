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
constexpr int kSOGMeshAssignOrder = 5;
constexpr int kSOGMeshAliasExtent = 8;

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

int wrap_index(int i, const int n) {
  i %= n;
  if (i < 0) {
    i += n;
  }
  return i;
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

void bspline_weights_1d(const double frac,
                        std::array<double, kSOGMeshAssignOrder> &w) {
  w.fill(0.0);
  w[0] = 1.0 - frac;
  w[1] = frac;
  for (int k = 3; k <= kSOGMeshAssignOrder; ++k) {
    const double inv = 1.0 / static_cast<double>(k - 1);
    w[static_cast<size_t>(k - 1)] = frac * w[static_cast<size_t>(k - 2)] * inv;
    for (int j = 1; j <= k - 2; ++j) {
      w[static_cast<size_t>(k - 1 - j)] =
          ((frac + static_cast<double>(j)) *
               w[static_cast<size_t>(k - 2 - j)] +
           (static_cast<double>(k - j) - frac) *
               w[static_cast<size_t>(k - 1 - j)]) *
          inv;
    }
    w[0] = (1.0 - frac) * w[0] * inv;
  }
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
  : KSpace(lmp),
      accuracy_in(1e-6),
      n_dl(1.0),
      remove_self_interaction(false),
      use_finufft(true),
      finufft_eps(kSOGDefaultFinufftEps),
      finufft_library(),
      finufft_warned(false),
  mesh_oversample(1.5),
  mesh_alias_extent(kSOGMeshAliasExtent),
      b_param(kSOGDefaultB),
      sigma_param(kSOGDefaultSigma),
      m_param(kSOGDefaultM),
      self_diag_sum(0.0),
      kernel_ready(false),
      mesh_ready(false),
      mesh_nx(0),
      mesh_ny(0),
      mesh_nz(0),
      mesh_lx(0.0),
      mesh_ly(0.0),
      mesh_lz(0.0),
      mesh_fft(nullptr) {
  triclinic_support = 0;
}

SOGKSpace::~SOGKSpace() { destroy_fft_plan(); }

bool SOGKSpace::is_keyword(const std::string &token) const {
  const std::string key = to_lower_copy(token);
  return key == "n_dl" || key == "remove_self_interaction" || key == "b" ||
         key == "sigma" || key == "m" || key == "amp" ||
         key == "bandwidth" || key == "use_finufft" ||
         key == "finufft_eps" || key == "finufft_library" ||
         key == "mesh_oversample" || key == "mesh_alias_extent";
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

  accuracy_relative = accuracy_in;

  n_dl = 1.0;
  remove_self_interaction = false;
  use_finufft = true;
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
  triclinic_check();

  if (domain->dimension == 2) {
    error->all(FLERR, "Cannot use kspace style sog with 2d simulation");
  }
  if (domain->triclinic != 0) {
    error->all(FLERR,
               "kspace style sog currently supports only orthorhombic boxes");
  }
  if (!atom->q_flag) {
    error->all(FLERR, "Kspace style sog requires atom attribute q");
  }
  if (domain->nonperiodic > 0 || domain->xperiodic != 1 ||
      domain->yperiodic != 1 || domain->zperiodic != 1) {
    error->all(FLERR,
               "kspace style sog currently requires fully periodic boundaries");
  }

  if (!kernel_ready) {
    finalize_kernel_parameters();
  }

  two_charge();
  pair_check();

  int itmp = 0;
  auto *p_cutoff = (double *) force->pair->extract("cut_coul", itmp);
  if (p_cutoff == nullptr) {
    error->all(FLERR,
               "kspace style sog is incompatible with current pair style");
  }

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  setup();
}

void SOGKSpace::setup() {
  ensure_fft_plan();
}

size_t SOGKSpace::mesh_index(int ix, int iy, int iz) const {
  return static_cast<size_t>(ix + mesh_nx * (iy + mesh_ny * iz));
}

double SOGKSpace::periodic_fraction(double x, double xlo, double prd) const {
  double frac = (x - xlo) / prd;
  frac -= std::floor(frac);
  return frac;
}

void SOGKSpace::destroy_fft_plan() {
  if (mesh_fft != nullptr) {
    delete mesh_fft;
    mesh_fft = nullptr;
  }
  mesh_ready = false;
  mesh_nx = 0;
  mesh_ny = 0;
  mesh_nz = 0;
  mesh_lx = mesh_ly = mesh_lz = 0.0;
  mesh_rho.clear();
  mesh_fft_work.clear();
  mesh_gradx.clear();
  mesh_grady.clear();
  mesh_gradz.clear();
}

void SOGKSpace::ensure_fft_plan() {
  if (!(domain->xprd > 0.0 && domain->yprd > 0.0 && domain->zprd > 0.0)) {
    error->all(FLERR,
               "kspace style sog encountered non-positive box length while "
               "building mesh FFT");
  }

  const double lx = domain->xprd;
  const double ly = domain->yprd;
  const double lz = domain->zprd;

    // Nyquist condition: pi / d >= 2*pi / n_dl  =>  d <= n_dl / 2.
    // mesh_oversample scales beyond the minimum Nyquist-compliant grid.
    const double mesh_scale = std::max(1.0, mesh_oversample);
    int nx =
      std::max(8, static_cast<int>(std::ceil(mesh_scale * 2.0 * lx / n_dl)));
    int ny =
      std::max(8, static_cast<int>(std::ceil(mesh_scale * 2.0 * ly / n_dl)));
    int nz =
      std::max(8, static_cast<int>(std::ceil(mesh_scale * 2.0 * lz / n_dl)));

  if (nx & 1) {
    ++nx;
  }
  if (ny & 1) {
    ++ny;
  }
  if (nz & 1) {
    ++nz;
  }

  if (mesh_ready && mesh_nx == nx && mesh_ny == ny && mesh_nz == nz &&
      std::fabs(mesh_lx - lx) < 1e-12 && std::fabs(mesh_ly - ly) < 1e-12 &&
      std::fabs(mesh_lz - lz) < 1e-12) {
    return;
  }

  destroy_fft_plan();

  const int64_t ngrid64 = static_cast<int64_t>(nx) * static_cast<int64_t>(ny) *
                          static_cast<int64_t>(nz);
  if (ngrid64 <= 0 ||
      ngrid64 > static_cast<int64_t>(std::numeric_limits<size_t>::max() / 2)) {
    error->all(FLERR, "kspace style sog mesh grid size overflow");
  }

  mesh_nx = nx;
  mesh_ny = ny;
  mesh_nz = nz;
  mesh_lx = lx;
  mesh_ly = ly;
  mesh_lz = lz;

  const size_t ngrid = static_cast<size_t>(ngrid64);
  mesh_rho.assign(ngrid, 0.0);
  mesh_fft_work.assign(2 * ngrid, 0.0);
  mesh_gradx.assign(2 * ngrid, 0.0);
  mesh_grady.assign(2 * ngrid, 0.0);
  mesh_gradz.assign(2 * ngrid, 0.0);

  int tmp = 0;
  mesh_fft = new FFT3d(lmp,
                       world,
                       mesh_nx,
                       mesh_ny,
                       mesh_nz,
                       0,
                       mesh_nx - 1,
                       0,
                       mesh_ny - 1,
                       0,
                       mesh_nz - 1,
                       0,
                       mesh_nx - 1,
                       0,
                       mesh_ny - 1,
                       0,
                       mesh_nz - 1,
                       0,
                       0,
                       &tmp,
                       collective_flag);

  mesh_ready = true;
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
          "available via FINUFFT/OMP_NUM_THREADS. Falling back to internal mesh "
          "FFT path "
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
    error->all(
        FLERR,
        "kspace style sog failed to load FINUFFT ({}). Configure "
        "finufft_library or DP_SOG_FINUFFT_LIBRARY before using FINUFFT "
        "mode",
        loader.error);
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
    error->all(FLERR,
               "kspace style sog encountered non-positive box length in "
               "FINUFFT path");
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
    error->all(FLERR,
               "kspace style sog FINUFFT mode grid is too large");
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
    error->all(FLERR,
               "kspace style sog FINUFFT makeplan(type1) failed with code {}",
               ier);
  }

  ier = loader.api.setpts(plan1, static_cast<int64_t>(nlocal), xj.data(), yj.data(),
                          zj.data(), 0, nullptr, nullptr, nullptr);
  if (ier == 0) {
    ier = loader.api.execute(plan1, q_complex.data(), rho_k.data());
  }
  loader.api.destroy(plan1);
  if (ier != 0) {
    error->all(FLERR,
               "kspace style sog FINUFFT type1 execution failed with code {}",
               ier);
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

        const double kfac = spectral_kernel(sqk);
        if (!std::isfinite(kfac) || kfac == 0.0) {
          continue;
        }

        const size_t idx =
            static_cast<size_t>(ix + ms * (iy + mt * iz));
        const std::complex<double> rho = rho_k[idx];
        // gaussian.py alignment:
        // E_recip = sum_k(kfac * |rho_k|^2) / (2 * volume)
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
    // gaussian.py alignment:
    // E_self = qsqsum * sum_k(kfac) / (2 * volume)
    energy_local -= qsqsum_local * (diag_sum / (2.0 * volume_local));
  }

  finufft_plan plan2 = nullptr;
  ier = loader.api.makeplan(2, 3, nmodes, 1, 1, finufft_eps, &plan2, nullptr);
  if (ier != 0 || plan2 == nullptr) {
    if (plan2 != nullptr) {
      loader.api.destroy(plan2);
    }
    error->all(FLERR,
               "kspace style sog FINUFFT makeplan(type2) failed with code {}",
               ier);
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
    error->all(FLERR,
               "kspace style sog FINUFFT type2 execution failed with code {}",
               ier);
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

void SOGKSpace::compute_mesh_fft(int eflag, int vflag) {
  if (domain->triclinic != 0) {
    error->all(FLERR,
               "kspace style sog mesh FFT path currently supports only "
               "orthorhombic boxes");
  }
  if (comm->nprocs != 1) {
    error->all(FLERR,
               "kspace style sog mesh FFT path currently supports only single "
               "MPI rank");
  }

  ensure_fft_plan();

  const bool want_energy_global = (eflag & ENERGY_GLOBAL);
  const bool want_virial_global = (vflag & (VIRIAL_PAIR | VIRIAL_FDOTR));

  energy = 0.0;
  for (int j = 0; j < 6; ++j) {
    virial[j] = 0.0;
  }

  std::fill(mesh_rho.begin(), mesh_rho.end(), 0.0);
  std::fill(mesh_fft_work.begin(), mesh_fft_work.end(), 0.0);
  std::fill(mesh_gradx.begin(), mesh_gradx.end(), 0.0);
  std::fill(mesh_grady.begin(), mesh_grady.end(), 0.0);
  std::fill(mesh_gradz.begin(), mesh_gradz.end(), 0.0);

  const int nlocal = atom->nlocal;
  if (nlocal <= 0) {
    return;
  }

  const double dx = mesh_lx / static_cast<double>(mesh_nx);
  const double dy = mesh_ly / static_cast<double>(mesh_ny);
  const double dz = mesh_lz / static_cast<double>(mesh_nz);
  const double volume_local = mesh_lx * mesh_ly * mesh_lz;
  const double rho_scale =
      static_cast<double>(mesh_nx * mesh_ny * mesh_nz) / volume_local;

  constexpr int assign_order = kSOGMeshAssignOrder;
  constexpr int assign_half = (kSOGMeshAssignOrder - 1) / 2;

  double **x = atom->x;
  double *q = atom->q;

  for (int i = 0; i < nlocal; ++i) {
    const double fx = periodic_fraction(x[i][0], domain->boxlo[0], mesh_lx) *
                      static_cast<double>(mesh_nx);
    const double fy = periodic_fraction(x[i][1], domain->boxlo[1], mesh_ly) *
                      static_cast<double>(mesh_ny);
    const double fz = periodic_fraction(x[i][2], domain->boxlo[2], mesh_lz) *
                      static_cast<double>(mesh_nz);

    const int ix0 = static_cast<int>(std::floor(fx));
    const int iy0 = static_cast<int>(std::floor(fy));
    const int iz0 = static_cast<int>(std::floor(fz));
    const double tx = fx - static_cast<double>(ix0);
    const double ty = fy - static_cast<double>(iy0);
    const double tz = fz - static_cast<double>(iz0);

    std::array<double, assign_order> wx;
    std::array<double, assign_order> wy;
    std::array<double, assign_order> wz;
    bspline_weights_1d(tx, wx);
    bspline_weights_1d(ty, wy);
    bspline_weights_1d(tz, wz);

    std::array<int, assign_order> ix;
    std::array<int, assign_order> iy;
    std::array<int, assign_order> iz;
    for (int a = 0; a < assign_order; ++a) {
      ix[static_cast<size_t>(a)] = wrap_index(ix0 - assign_half + a, mesh_nx);
      iy[static_cast<size_t>(a)] = wrap_index(iy0 - assign_half + a, mesh_ny);
      iz[static_cast<size_t>(a)] = wrap_index(iz0 - assign_half + a, mesh_nz);
    }

    for (int a = 0; a < assign_order; ++a) {
      for (int b = 0; b < assign_order; ++b) {
        for (int c = 0; c < assign_order; ++c) {
          const size_t idx = mesh_index(ix[a], iy[b], iz[c]);
          mesh_rho[idx] +=
              static_cast<FFT_SCALAR>(rho_scale * q[i] * wx[a] * wy[b] * wz[c]);
        }
      }
    }
  }

  const size_t ngrid = mesh_rho.size();
  for (size_t idx = 0; idx < ngrid; ++idx) {
    mesh_fft_work[2 * idx] = mesh_rho[idx];
    mesh_fft_work[2 * idx + 1] = 0.0;
  }

  mesh_fft->compute(mesh_fft_work.data(), mesh_fft_work.data(), FFT3d::FORWARD);

  const double scaleinv = 1.0 / static_cast<double>(ngrid);
  const double s2 = scaleinv * scaleinv;
  const double k_sq_max = (MY_2PI / n_dl) * (MY_2PI / n_dl);
  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;
  const int alias_extent = mesh_alias_extent;
  const int assign_pow = 2 * assign_order;

  double energy_local = 0.0;
  double diag_sum_local = 0.0;

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);
    double sz_sum = 0.0;
    for (int jz = -alias_extent; jz <= alias_extent; ++jz) {
      const double qz = twopi_over_z * static_cast<double>(kz_mode + mesh_nz * jz);
      sz_sum += sinc_pow(0.5 * qz * dz, assign_pow);
    }

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);
      double sy_sum = 0.0;
      for (int jy = -alias_extent; jy <= alias_extent; ++jy) {
        const double qy =
            twopi_over_y * static_cast<double>(ky_mode + mesh_ny * jy);
        sy_sum += sinc_pow(0.5 * qy * dy, assign_pow);
      }

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);
        const double w2x = sinc_pow(0.5 * kx * dx, assign_pow);
        double sx_sum = 0.0;
        for (int jx = -alias_extent; jx <= alias_extent; ++jx) {
          const double qx =
              twopi_over_x * static_cast<double>(kx_mode + mesh_nx * jx);
          sx_sum += sinc_pow(0.5 * qx * dx, assign_pow);
        }

        const size_t idx = mesh_index(ix, iy, iz);
        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        const double kfac = spectral_kernel(sqk);
        if (!std::isfinite(kfac) || kfac == 0.0) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        const double denom_lin = sx_sum * sy_sum * sz_sum;
        const double denominator = denom_lin * denom_lin;
        if (!(denominator > 1e-20) || !std::isfinite(denominator)) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        double sum0 = 0.0;
        double sum1 = 0.0;
        for (int jx = -alias_extent; jx <= alias_extent; ++jx) {
          const double qx =
              twopi_over_x * static_cast<double>(kx_mode + mesh_nx * jx);
          const double wx_alias = sinc_pow(0.5 * qx * dx, assign_pow);

          for (int jy = -alias_extent; jy <= alias_extent; ++jy) {
            const double qy =
                twopi_over_y * static_cast<double>(ky_mode + mesh_ny * jy);
            const double wy_alias = sinc_pow(0.5 * qy * dy, assign_pow);

            for (int jz = -alias_extent; jz <= alias_extent; ++jz) {
              const double qz =
                  twopi_over_z * static_cast<double>(kz_mode + mesh_nz * jz);
              const double wz_alias = sinc_pow(0.5 * qz * dz, assign_pow);

              const double qsq = qx * qx + qy * qy + qz * qz;
              if (!(qsq > 0.0 && qsq <= k_sq_max)) {
                continue;
              }

              const double kfac_alias = spectral_kernel(qsq);
              if (!std::isfinite(kfac_alias) || kfac_alias == 0.0) {
                continue;
              }

              sum0 += kfac_alias * wx_alias * wy_alias * wz_alias;
              const double dot1 = kx * qx + ky * qy + kz * qz;
              sum1 += dot1 * kfac_alias * wx_alias * wy_alias * wz_alias;
            }
          }
        }

        const double geff_energy = sum0 / denominator;
        const double geff = sum1 / (sqk * denominator);
        if (!std::isfinite(geff_energy) || !std::isfinite(geff)) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        const double w2y = sinc_pow(0.5 * ky * dy, assign_pow);
        const double w2z = sinc_pow(0.5 * kz * dz, assign_pow);
        const double w2_principal = w2x * w2y * w2z;
        if (w2_principal > 1e-20) {
          diag_sum_local += sum0 / w2_principal;
        }

        const double rho_re = static_cast<double>(mesh_fft_work[2 * idx]);
        const double rho_im = static_cast<double>(mesh_fft_work[2 * idx + 1]);

        if (want_energy_global) {
          energy_local +=
              s2 * geff_energy * (rho_re * rho_re + rho_im * rho_im);
        }

        const double vk_re = scaleinv * geff * rho_re;
        const double vk_im = scaleinv * geff * rho_im;

        mesh_gradx[2 * idx] = static_cast<FFT_SCALAR>(-kx * vk_im);
        mesh_gradx[2 * idx + 1] = static_cast<FFT_SCALAR>(kx * vk_re);
        mesh_grady[2 * idx] = static_cast<FFT_SCALAR>(-ky * vk_im);
        mesh_grady[2 * idx + 1] = static_cast<FFT_SCALAR>(ky * vk_re);
        mesh_gradz[2 * idx] = static_cast<FFT_SCALAR>(-kz * vk_im);
        mesh_gradz[2 * idx + 1] = static_cast<FFT_SCALAR>(kz * vk_re);
      }
    }
  }

  mesh_fft->compute(mesh_gradx.data(), mesh_gradx.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_grady.data(), mesh_grady.data(), FFT3d::BACKWARD);
  mesh_fft->compute(mesh_gradz.data(), mesh_gradz.data(), FFT3d::BACKWARD);

  const double qscale_local = force->qqrd2e * scale;
  std::array<double, 6> virial_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < nlocal; ++i) {
    const double fx = periodic_fraction(x[i][0], domain->boxlo[0], mesh_lx) *
                      static_cast<double>(mesh_nx);
    const double fy = periodic_fraction(x[i][1], domain->boxlo[1], mesh_ly) *
                      static_cast<double>(mesh_ny);
    const double fz = periodic_fraction(x[i][2], domain->boxlo[2], mesh_lz) *
                      static_cast<double>(mesh_nz);

    const int ix0 = static_cast<int>(std::floor(fx));
    const int iy0 = static_cast<int>(std::floor(fy));
    const int iz0 = static_cast<int>(std::floor(fz));
    const double tx = fx - static_cast<double>(ix0);
    const double ty = fy - static_cast<double>(iy0);
    const double tz = fz - static_cast<double>(iz0);

    std::array<double, assign_order> wx;
    std::array<double, assign_order> wy;
    std::array<double, assign_order> wz;
    bspline_weights_1d(tx, wx);
    bspline_weights_1d(ty, wy);
    bspline_weights_1d(tz, wz);

    std::array<int, assign_order> ix;
    std::array<int, assign_order> iy;
    std::array<int, assign_order> iz;
    for (int a = 0; a < assign_order; ++a) {
      ix[static_cast<size_t>(a)] = wrap_index(ix0 - assign_half + a, mesh_nx);
      iy[static_cast<size_t>(a)] = wrap_index(iy0 - assign_half + a, mesh_ny);
      iz[static_cast<size_t>(a)] = wrap_index(iz0 - assign_half + a, mesh_nz);
    }

    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
    for (int a = 0; a < assign_order; ++a) {
      for (int b = 0; b < assign_order; ++b) {
        for (int c = 0; c < assign_order; ++c) {
          const size_t idx = mesh_index(ix[a], iy[b], iz[c]);
          const double w = wx[a] * wy[b] * wz[c];
          gx += w * static_cast<double>(mesh_gradx[2 * idx]);
          gy += w * static_cast<double>(mesh_grady[2 * idx]);
          gz += w * static_cast<double>(mesh_gradz[2 * idx]);
        }
      }
    }

    const double qi = q[i];
    const double fxs = -qscale_local * qi * gx;
    const double fys = -qscale_local * qi * gy;
    const double fzs = -qscale_local * qi * gz;

    atom->f[i][0] += fxs;
    atom->f[i][1] += fys;
    atom->f[i][2] += fzs;

    if (want_virial_global) {
      virial_local[0] += x[i][0] * fxs;
      virial_local[1] += x[i][1] * fys;
      virial_local[2] += x[i][2] * fzs;
      virial_local[3] += x[i][0] * fys;
      virial_local[4] += x[i][0] * fzs;
      virial_local[5] += x[i][1] * fzs;
    }
  }

  if (want_energy_global) {
    double energy_all = 0.0;
    MPI_Allreduce(&energy_local, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);

    double diag_sum_all = 0.0;
    MPI_Allreduce(&diag_sum_local, &diag_sum_all, 1, MPI_DOUBLE, MPI_SUM, world);

    double qsqsum_local = 0.0;
    for (int i = 0; i < nlocal; ++i) {
      qsqsum_local += q[i] * q[i];
    }
    double qsqsum_all = 0.0;
    MPI_Allreduce(&qsqsum_local, &qsqsum_all, 1, MPI_DOUBLE, MPI_SUM, world);

    self_diag_sum = diag_sum_all / (2.0 * volume_local);

    energy = 0.5 * volume_local * energy_all;
    if (remove_self_interaction) {
      energy -= qsqsum_all * self_diag_sum;
    }
    energy *= qscale_local;
  }

  if (want_virial_global) {
    double virial_all[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(virial_local.data(), virial_all, 6, MPI_DOUBLE, MPI_SUM, world);
    for (int j = 0; j < 6; ++j) {
      virial[j] = virial_all[j];
    }
  }
}

void SOGKSpace::compute(int eflag, int vflag) {
  ev_init(eflag, vflag, 0);

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  if (qsqsum == 0.0) {
    energy = 0.0;
    for (int j = 0; j < 6; ++j) {
      virial[j] = 0.0;
    }
    return;
  }

  if (compute_finufft(eflag, vflag)) {
    return;
  }

  compute_mesh_fft(eflag, vflag);
}

double SOGKSpace::memory_usage() {
  const size_t mesh_bytes =
      mesh_rho.capacity() * sizeof(FFT_SCALAR) +
      mesh_fft_work.capacity() * sizeof(FFT_SCALAR) +
      mesh_gradx.capacity() * sizeof(FFT_SCALAR) +
      mesh_grady.capacity() * sizeof(FFT_SCALAR) +
      mesh_gradz.capacity() * sizeof(FFT_SCALAR);
  const size_t param_bytes =
      (amp.capacity() + bandwidth.capacity()) * sizeof(double);
  return static_cast<double>(mesh_bytes + param_bytes);
}
