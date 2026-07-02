// SPDX-License-Identifier: LGPL-3.0-or-later

#include "sog.h"
#include "sog_spline.h"

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

constexpr int kSOGGridMin = 8;
constexpr int kSOGGridMaxIter = 500;

bool factorable_235(int n) {
  while (n > 1) {
    if ((n % 2) == 0) {
      n /= 2;
    } else if ((n % 3) == 0) {
      n /= 3;
    } else if ((n % 5) == 0) {
      n /= 5;
    } else {
      return false;
    }
  }
  return true;
}

double pppm_ik_error_estimate_order5(const double h,
                                     const double prd,
                                     const bigint natoms,
                                     const double q2,
                                     const double g_eff) {
  if (!(natoms > 0) || !(h > 0.0) || !(prd > 0.0) || !(q2 > 0.0) ||
      !(g_eff > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }

  // PPPM order-5 acons coefficients (same error model family as PPPM).
  static constexpr double acons_order5[] = {
      1.0 / 23232.0,
      7601.0 / 13628160.0,
      143.0 / 69120.0,
      517231.0 / 106536960.0,
      106640677.0 / 11737571328.0,
  };

  double series = 0.0;
  for (int m = 0; m < 5; ++m) {
    series +=
        acons_order5[m] * std::pow(h * g_eff, 2.0 * static_cast<double>(m));
  }

  const double prefactor = q2 * std::pow(h * g_eff, 5.0);
  const double root = std::sqrt(g_eff * prd * std::sqrt(MY_2PI) * series /
                                static_cast<double>(natoms));
  return prefactor * root / (prd * prd);
}

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

// ── 1D analytic Fourier integrals for the CubeS₂ influence function ──
// I_p(α) = ∫₀¹ t^p · exp(i·α·t) dt
// Recurrence-free closed forms with Taylor expansions for |α| < 1e-8.
// Mirrors fastsog.cpp (used to build |Φ(k)|² analytically).

inline std::complex<double> sog_I_int_0(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    return std::complex<double>(1.0 - alpha * alpha / 6.0,
                                alpha / 2.0 - alpha * alpha * alpha / 24.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  return std::complex<double>(sin_a / alpha, (1.0 - cos_a) / alpha);
}

inline std::complex<double> sog_I_int_1(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    return std::complex<double>(0.5 - alpha * alpha / 8.0,
                                alpha / 3.0 - alpha * alpha * alpha / 30.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  const double a2 = alpha * alpha;
  return std::complex<double>((alpha * sin_a + cos_a - 1.0) / a2,
                              (sin_a - alpha * cos_a) / a2);
}

inline std::complex<double> sog_I_int_2(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    return std::complex<double>(1.0 / 3.0 - alpha * alpha / 10.0, alpha / 4.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  const double a2 = alpha * alpha;
  const double a3 = a2 * alpha;
  return std::complex<double>(
      (2.0 * alpha * sin_a + (a2 - 2.0) * cos_a + 2.0) / a3,
      ((a2 - 2.0) * sin_a + 2.0 * alpha * cos_a) / a3);
}

inline std::complex<double> sog_I_int_3(const double alpha) {
  if (std::fabs(alpha) < 1e-8) {
    return std::complex<double>(0.25, alpha / 5.0);
  }
  const double cos_a = std::cos(alpha);
  const double sin_a = std::sin(alpha);
  const double a2 = alpha * alpha;
  const double a3 = a2 * alpha;
  const double a4 = a3 * alpha;
  return std::complex<double>(
      ((3.0 * a2 - 6.0) * alpha * sin_a + (a3 - 6.0 * alpha) * cos_a + 6.0 * alpha) / a4,
      ((a3 - 6.0 * alpha) * sin_a + (6.0 - 3.0 * a2) * cos_a + 3.0 * a2 - 6.0) / a4);
}

// ── Monomial expansion for CubeS₂ 4th-order node weights ──
// Each entry: (pow_x, pow_y, pow_z, real_coeff). Mirrors fastsog.cpp.
struct SogMonomialTerm {
  int px, py, pz;
  double coeff;
};

constexpr int kSogMaxMonomialsPerNode = 64;

struct SogCubeS2NodeMonomial {
  int num_terms;
  SogMonomialTerm terms[kSogMaxMonomialsPerNode];
};

// Expand the CubeS₂ weight c_d(θ) into monomials θ_x^px · θ_y^py · θ_z^pz so the
// influence function Φ(k) = Σ_d e^{i k·d·Δ} Σ C·I_px(αx)·I_py(αy)·I_pz(αz) can be
// evaluated from the precomputed 1D integrals I_p.
inline void sog_build_monomials_for_node(const SogCubeS2Node4 &node, const double xi,
                                         SogCubeS2NodeMonomial &result) {
  result.num_terms = 0;
  const int dx = node.dx, dy = node.dy, dz = node.dz;
  const double a[3] = {static_cast<double>(dx),
                        static_cast<double>(dy),
                        static_cast<double>(dz)};
  const double b[3] = {1.0 - 2.0 * a[0],
                        1.0 - 2.0 * a[1],
                        1.0 - 2.0 * a[2]};

  const double xi2 = xi * xi;

  auto binom = [](int n, int k) -> double {
    if (k < 0 || k > n) return 0.0;
    constexpr double C[4][4] = {
      {1, 0, 0, 0},
      {1, 1, 0, 0},
      {1, 2, 1, 0},
      {1, 3, 3, 1},
    };
    return C[n][k];
  };

  if (node.cls == 0) {
    // Class 0: c_d = L(ηx)·ηy·ηz + L(ηy)·ηz·ηx + L(ηz)·ηx·ηy
    const double xi2_adj = (9.0 * xi2 - 2.0) / 6.0;
    const double L_coeffs[4] = {0.5 * xi2, -xi2_adj, 0.5, -0.5};

    for (int term_idx = 0; term_idx < 3; ++term_idx) {
      int axis_L = term_idx;
      int axis_n1 = (term_idx + 1) % 3;
      int axis_n2 = (term_idx + 2) % 3;

      for (int pL = 0; pL <= 3; ++pL) {
        const double c_L = L_coeffs[pL];
        if (c_L == 0.0) continue;
        for (int jL = 0; jL <= pL; ++jL) {
          const double cf_L = c_L * binom(pL, jL) *
            std::pow(a[axis_L], static_cast<double>(pL - jL)) *
            std::pow(b[axis_L], static_cast<double>(jL));
          for (int jn1 = 0; jn1 <= 1; ++jn1) {
            const double cf_n1 = binom(1, jn1) *
              std::pow(a[axis_n1], static_cast<double>(1 - jn1)) *
              std::pow(b[axis_n1], static_cast<double>(jn1));
            for (int jn2 = 0; jn2 <= 1; ++jn2) {
              const double cf_n2 = binom(1, jn2) *
                std::pow(a[axis_n2], static_cast<double>(1 - jn2)) *
                std::pow(b[axis_n2], static_cast<double>(jn2));
              const double coeff = cf_L * cf_n1 * cf_n2;
              if (coeff == 0.0) continue;
              int pows[3] = {0, 0, 0};
              pows[axis_L] = jL;
              pows[axis_n1] = jn1;
              pows[axis_n2] = jn2;
              bool merged = false;
              for (int m = 0; m < result.num_terms; ++m) {
                if (result.terms[m].px == pows[0] &&
                    result.terms[m].py == pows[1] &&
                    result.terms[m].pz == pows[2]) {
                  result.terms[m].coeff += coeff;
                  merged = true;
                  break;
                }
              }
              if (!merged && result.num_terms < kSogMaxMonomialsPerNode) {
                result.terms[result.num_terms] = {pows[0], pows[1], pows[2], coeff};
                result.num_terms++;
              }
            }
          }
        }
      }
    }
  } else {
    // Class 1: c_d = R(η_special) · η_n1 · η_n2  (single term, paper Eq. 16)
    // R(t) = ⅙·t³ + (3ξ²-1)/6·t  →  coeffs [1/6, 0, (3ξ²-1)/6, 0]
    const double R_coeffs[4] = {0.0, (3.0 * xi2 - 1.0) / 6.0, 0.0, 1.0 / 6.0};
    int axis_L = node.sp_axis;
    int axis_n1 = (axis_L + 1) % 3;
    int axis_n2 = (axis_L + 2) % 3;

    for (int pL = 0; pL <= 3; ++pL) {
      const double c_R = R_coeffs[pL];
      if (c_R == 0.0) continue;
      for (int jL = 0; jL <= pL; ++jL) {
        const double cf_L = c_R * binom(pL, jL) *
          std::pow(a[axis_L], static_cast<double>(pL - jL)) *
          std::pow(b[axis_L], static_cast<double>(jL));
        for (int jn1 = 0; jn1 <= 1; ++jn1) {
          const double cf_n1 = binom(1, jn1) *
            std::pow(a[axis_n1], static_cast<double>(1 - jn1)) *
            std::pow(b[axis_n1], static_cast<double>(jn1));
          for (int jn2 = 0; jn2 <= 1; ++jn2) {
            const double cf_n2 = binom(1, jn2) *
              std::pow(a[axis_n2], static_cast<double>(1 - jn2)) *
              std::pow(b[axis_n2], static_cast<double>(jn2));
            const double coeff = cf_L * cf_n1 * cf_n2;
            if (coeff == 0.0) continue;
            int pows[3] = {0, 0, 0};
            pows[axis_L] = jL;
            pows[axis_n1] = jn1;
            pows[axis_n2] = jn2;
            bool merged = false;
            for (int m = 0; m < result.num_terms; ++m) {
              if (result.terms[m].px == pows[0] &&
                  result.terms[m].py == pows[1] &&
                  result.terms[m].pz == pows[2]) {
                result.terms[m].coeff += coeff;
                merged = true;
                break;
              }
            }
            if (!merged && result.num_terms < kSogMaxMonomialsPerNode) {
              result.terms[result.num_terms] = {pows[0], pows[1], pows[2], coeff};
              result.num_terms++;
            }
          }
        }
      }
    }
  }
}

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
      spline_type(4),  // CubeS₂ 4th-order Midtown splines (default)
      b_param(kSOGDefaultB),
      sigma_param(kSOGDefaultSigma),
      m_param(kSOGDefaultM),
      self_diag_sum(0.0),
      self_coeff(0.0),
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
  return key == "n_dl" || key == "cubes2_phi_max" ||
         key == "remove_self_interaction" || key == "b" ||
         key == "sigma" || key == "m" || key == "amp" ||
         key == "bandwidth" || key == "use_finufft" ||
         key == "finufft_eps" || key == "finufft_library" ||
         key == "mesh_oversample" || key == "mesh_alias_extent" ||
         key == "spline";
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

// ── RBSOG self-energy helpers ──

static double G_sigma(const double sigma, const double r) {
  return std::exp(-r * r / (2.0 * sigma * sigma)) /
         std::sqrt(2.0 * MY_PI * sigma * sigma);
}

static double compute_w0(const double r0, const double b) {
  // r0 = rcut / sigma
  double sum = 0.0;
  for (int i = 1; i < 200; ++i) {
    const double bi = std::pow(b, static_cast<double>(-i));
    sum += bi * G_sigma(1.0, bi * r0);
  }
  return (1.0 / G_sigma(1.0, r0)) *
         ((1.0 / (2.0 * std::log(b) * r0)) - sum);
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
    amp.resize(bandwidth.size());
    for (size_t mm = 0; mm < bandwidth.size(); ++mm) {
      amp[mm] = amp0 * bandwidth[mm];
    }
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

double SOGKSpace::virial_kernel(const double sqk) const {
  // ∂K/∂(k²) weighted by bandwidth:
  //   virial_kernel(k²) = Σ_ℓ  amp[ℓ] · bandwidth[ℓ] · exp(-½ bandwidth[ℓ] · k²)
  // Used to build the k-space virial via (rbsog-npt.md §3.1):
  //   W_{αβ}^F = (1/(2V)) Σ_k |ρ(k)|² [K(k²)δ_{αβ} - virial_kernel(k²)·k_α·k_β]
  if (!(sqk > 0.0)) {
    return 0.0;
  }

  double coeff = 0.0;
  for (size_t mm = 0; mm < amp.size(); ++mm) {
    coeff += amp[mm] * bandwidth[mm] * std::exp(-0.5 * bandwidth[mm] * sqk);
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
  cubes2_phi_max = 0.0;  // 0 = auto from Table III
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
  spline_type = 0;

  int iarg = 1;
  while (iarg < narg) {
    const std::string key = to_lower_copy(arg[iarg]);
    if (key == "n_dl") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing n_dl value");
      }
      n_dl = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "cubes2_phi_max") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing cubes2_phi_max value");
      }
      cubes2_phi_max = atof(arg[iarg + 1]);
      if (!(std::isfinite(cubes2_phi_max) && cubes2_phi_max > 0.0)) {
        error->all(FLERR,
                   "kspace style sog cubes2_phi_max must be positive finite");
      }
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
    } else if (key == "spline") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style sog missing spline value");
      }
      const std::string val = to_lower_copy(arg[iarg + 1]);
      if (val == "bspline") {
        spline_type = 0;
      } else if (val == "cubes2_4") {
        spline_type = 4;
      } else if (val == "cubes2_6") {
        spline_type = 6;
      } else {
        error->all(FLERR,
                   "kspace style sog spline expects bspline, cubes2_4, or cubes2_6");
      }
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
  const double rcut = *p_cutoff;
  if (!(rcut > 0.0)) {
    error->all(FLERR, "kspace style sog requires positive cut_coul");
  }

  // Compute self-coefficient for RBSOG real-space self-energy correction.
  // self_coeff = log(b) / (sqrt(2π)·σ) · (w0 + Σ_{m=1}^{M-1} b^{-m})
  // where w0 enforces continuity of the u-series at r=rcut.
  const double r0 = rcut / sigma_param;
  const double w0 = compute_w0(r0, b_param);
  const double logb = std::log(b_param);
  double sum_b = 0.0;
  for (int m = 1; m < m_param; ++m) {
    sum_b += std::pow(b_param, static_cast<double>(-m));
  }
  self_coeff = (logb / (std::sqrt(2.0 * MY_PI) * sigma_param)) *
               (w0 + sum_b);

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
  mesh_green_energy.clear();
  mesh_green_force.clear();
  mesh_green_self.clear();
  sinc_table_x.clear();
  sinc_table_y.clear();
  sinc_table_z.clear();
  sinc_sum_x.clear();
  sinc_sum_y.clear();
  sinc_sum_z.clear();
  cubes2_influence_re.clear();
  cubes2_influence_im.clear();
  cubes2_influence_sq.clear();
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

  // Retrieve rcut for grid sizing (needed by both B-spline PPPM and CubeS₂
  // SOG-bandwidth methods).
  double rcut = n_dl;  // fallback
  {
    int itmp = 0;
    auto *p_cutoff = (double *) force->pair->extract("cut_coul", itmp);
    if (p_cutoff != nullptr && *p_cutoff > 0.0) {
      rcut = *p_cutoff;
    }
  }

  int nx, ny, nz;

  if (spline_type >= 4) {
    // SOG-bandwidth grid estimation for CubeS₂ Midtown splines.
    // φ = Δ/r_c from Predescu 2020 Table III (or user-specified).
    double phi_val;
    if (cubes2_phi_max > 0.0) {
      // User-specified φ value
      phi_val = cubes2_phi_max;
    } else {
      // Auto from Table III (Predescu 2020 JCP 153, 224117):
      //   CubeS₂ 4th, b=2:      φ_max = 0.23
      //   CubeS₂ 4th, b≈1.630:  φ_max = 0.065
      //   CubeS₂ 6th, b=2:      φ_max = 0.35
      //   CubeS₂ 6th, b≈1.630:  φ_max = 0.160
      // Linear interpolation between tabulated values.
      const double b_ref_lo = 1.6297670882677647;
      const double phi_lo = (spline_type >= 6) ? 0.160 : 0.065;
      const double b_ref_hi = 2.0;
      const double phi_hi = (spline_type >= 6) ? 0.350 : 0.230;
      double phi_max;
      if (b_param <= b_ref_lo) {
        phi_max = phi_lo;
      } else if (b_param >= b_ref_hi) {
        phi_max = phi_hi;
      } else {
        phi_max = phi_lo + (b_param - b_ref_lo) / (b_ref_hi - b_ref_lo)
                                * (phi_hi - phi_lo);
      }
      phi_val = phi_max;
    }
    const double delta = phi_val * rcut;
    nx = std::max(kSOGGridMin, static_cast<int>(std::ceil(lx / delta)));
    ny = std::max(kSOGGridMin, static_cast<int>(std::ceil(ly / delta)));
    nz = std::max(kSOGGridMin, static_cast<int>(std::ceil(lz / delta)));
    // PPPM refinement skipped — CubeS₂ uses φ-based grid directly
  } else {
    // Legacy B-spline grid: Nyquist lower bound from n_dl + PPPM refinement.
    const double mesh_scale = std::max(1.0, mesh_oversample);
    nx = std::max(
        kSOGGridMin, static_cast<int>(std::ceil(mesh_scale * 2.0 * lx / n_dl)));
    ny = std::max(
        kSOGGridMin, static_cast<int>(std::ceil(mesh_scale * 2.0 * ly / n_dl)));
    nz = std::max(
        kSOGGridMin, static_cast<int>(std::ceil(mesh_scale * 2.0 * lz / n_dl)));

    // PPPM-style refinement: use accuracy target to tighten grid counts.
    // Only run during initial mesh build (mesh_ready == false); on subsequent
    // calls the grid is kept fixed to avoid q2-dependent oscillations that
    // would trigger expensive mesh rebuilds every step.
    if (!mesh_ready && accuracy_in > 0.0 && q2 > 0.0 && atom->natoms > 0) {
      double cutoff = rcut;

      const double volume = lx * ly * lz;
      const double natoms = static_cast<double>(atom->natoms);

      double g_eff =
          accuracy_in * std::sqrt(natoms * cutoff * volume) / (2.0 * q2);
      if (!(g_eff > 0.0) || !std::isfinite(g_eff)) {
        g_eff = MY_2PI / n_dl;
      } else if (g_eff >= 1.0) {
        g_eff = (1.35 - 0.15 * std::log(accuracy_in)) / cutoff;
      } else {
        g_eff = std::sqrt(-std::log(g_eff)) / cutoff;
      }

      if (g_eff > 0.0 && std::isfinite(g_eff)) {
        double hx = 4.0 / g_eff;
        double hy = 4.0 / g_eff;
        double hz = 4.0 / g_eff;

        int nx_pppm = std::max(2, static_cast<int>(lx / hx));
        int ny_pppm = std::max(2, static_cast<int>(ly / hy));
        int nz_pppm = std::max(2, static_cast<int>(lz / hz));

        int count = 0;
        while (true) {
          const double errx =
              pppm_ik_error_estimate_order5(hx, lx, atom->natoms, q2, g_eff);
          const double erry =
              pppm_ik_error_estimate_order5(hy, ly, atom->natoms, q2, g_eff);
          const double errz =
              pppm_ik_error_estimate_order5(hz, lz, atom->natoms, q2, g_eff);
          const double err = std::max(errx, std::max(erry, errz));

          ++count;
          if (err <= accuracy_in) {
            break;
          }
          if (count > kSOGGridMaxIter) {
            break;
          }

          hx *= 0.95;
          hy *= 0.95;
          hz *= 0.95;
          nx_pppm = std::max(2, static_cast<int>(lx / hx));
          ny_pppm = std::max(2, static_cast<int>(ly / hy));
          nz_pppm = std::max(2, static_cast<int>(lz / hz));
        }

        nx = std::max(nx, nx_pppm);
        ny = std::max(ny, ny_pppm);
        nz = std::max(nz, nz_pppm);
      }
    }
  }

  while (!factorable_235(nx)) {
    ++nx;
  }
  while (!factorable_235(ny)) {
    ++ny;
  }
  while (!factorable_235(nz)) {
    ++nz;
  }

  // ── Case 1: Mesh already built, nothing changed ──
  if (mesh_ready && mesh_nx == nx && mesh_ny == ny && mesh_nz == nz &&
      std::fabs(mesh_lx - lx) < 1e-12 && std::fabs(mesh_ly - ly) < 1e-12 &&
      std::fabs(mesh_lz - lz) < 1e-12) {
    return;
  }

  // ── Case 2: Mesh already built, box changed but grid count unchanged ──
  // PPPM-style: keep the same FFT grid count, only recompute the Green
  // functions with the new box dimensions (spacing changed, k-vectors changed).
  // This avoids the expensive FFT plan rebuild on every NPT step.
  if (mesh_ready && mesh_nx == nx && mesh_ny == ny && mesh_nz == nz) {
    mesh_lx = lx;
    mesh_ly = ly;
    mesh_lz = lz;
    precompute_green_functions();
    return;
  }

  // ── Case 3: First build or grid count changed ──
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
                       collective_flag
#if LAMMPS_VERSION_NUMBER >= 20260330
                       ,
                       0
#endif
                       );

  if (spline_type >= 4) {
    precompute_cubes2_influence();
  } else {
    precompute_sinc_tables();
  }
  precompute_green_functions();

  mesh_ready = true;
}

void SOGKSpace::precompute_sinc_tables() {
  // Precompute box-independent sinc_pow values for alias sums.
  // sinc_pow(0.5 * q * dx, assign_pow) = sinc_pow(pi * (k_mode/n + j), assign_pow)
  // depends ONLY on grid indices, not on absolute box dimensions.
  //
  // This is called once during mesh creation (Case 3 of ensure_fft_plan).
  const int assign_pow = 2 * kSOGMeshAssignOrder;   // = 10
  const int alias_cnt = 2 * mesh_alias_extent + 1;   // = 17 for default extent

  // ── X dimension ──
  sinc_table_x.assign(static_cast<size_t>(mesh_nx) * static_cast<size_t>(alias_cnt),
                      0.0);
  sinc_sum_x.assign(static_cast<size_t>(mesh_nx), 0.0);
  for (int ix = 0; ix < mesh_nx; ++ix) {
    const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
    const double arg_base =
        MY_PI * static_cast<double>(kx_mode) / static_cast<double>(mesh_nx);
    const size_t base = static_cast<size_t>(ix) * static_cast<size_t>(alias_cnt);
    double sum = 0.0;
    for (int jx = -mesh_alias_extent; jx <= mesh_alias_extent; ++jx) {
      const double arg = arg_base + MY_PI * static_cast<double>(jx);
      const double val = sinc_pow(arg, assign_pow);
      sinc_table_x[base + static_cast<size_t>(jx + mesh_alias_extent)] = val;
      sum += val;
    }
    sinc_sum_x[static_cast<size_t>(ix)] = sum;
  }

  // ── Y dimension ──
  sinc_table_y.assign(static_cast<size_t>(mesh_ny) * static_cast<size_t>(alias_cnt),
                      0.0);
  sinc_sum_y.assign(static_cast<size_t>(mesh_ny), 0.0);
  for (int iy = 0; iy < mesh_ny; ++iy) {
    const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
    const double arg_base =
        MY_PI * static_cast<double>(ky_mode) / static_cast<double>(mesh_ny);
    const size_t base = static_cast<size_t>(iy) * static_cast<size_t>(alias_cnt);
    double sum = 0.0;
    for (int jy = -mesh_alias_extent; jy <= mesh_alias_extent; ++jy) {
      const double arg = arg_base + MY_PI * static_cast<double>(jy);
      const double val = sinc_pow(arg, assign_pow);
      sinc_table_y[base + static_cast<size_t>(jy + mesh_alias_extent)] = val;
      sum += val;
    }
    sinc_sum_y[static_cast<size_t>(iy)] = sum;
  }

  // ── Z dimension ──
  sinc_table_z.assign(static_cast<size_t>(mesh_nz) * static_cast<size_t>(alias_cnt),
                      0.0);
  sinc_sum_z.assign(static_cast<size_t>(mesh_nz), 0.0);
  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double arg_base =
        MY_PI * static_cast<double>(kz_mode) / static_cast<double>(mesh_nz);
    const size_t base = static_cast<size_t>(iz) * static_cast<size_t>(alias_cnt);
    double sum = 0.0;
    for (int jz = -mesh_alias_extent; jz <= mesh_alias_extent; ++jz) {
      const double arg = arg_base + MY_PI * static_cast<double>(jz);
      const double val = sinc_pow(arg, assign_pow);
      sinc_table_z[base + static_cast<size_t>(jz + mesh_alias_extent)] = val;
      sum += val;
    }
    sinc_sum_z[static_cast<size_t>(iz)] = sum;
  }
}

void SOGKSpace::precompute_cubes2_influence() {
  // Analytic CubeS₂ influence function Φ(k) for the 4th-order Midtown spline.
  // Φ(k) = Σ_d exp(i·k·d·Δ) · Σ_{a,b,c} C_d(a,b,c) · I_a(αx)·I_b(αy)·I_c(αz)
  // where I_p(α)=∫₀¹ tᵖ e^{iαt}dt is the 1D Fourier integral of the monomial
  // basis and C_d is the node weight polynomial expanded by
  // sog_build_monomials_for_node. The squared modulus |Φ(k)|² is the
  // assignment-function deconvolution denominator used by the CubeS₂ Green
  // function branch in precompute_green_functions().
  //
  // One-time cost per mesh build. Identical algorithm to fastsog.cpp
  // precompute_cubes2_influence() (verified correct against PPPM).
  const size_t ngrid = static_cast<size_t>(mesh_nx) *
                       static_cast<size_t>(mesh_ny) *
                       static_cast<size_t>(mesh_nz);

  cubes2_influence_re.assign(ngrid, 0.0);
  cubes2_influence_im.assign(ngrid, 0.0);
  cubes2_influence_sq.assign(ngrid, 0.0);

  auto *node_mono = new SogCubeS2NodeMonomial[kSogCubes2NumNodes4];
  const double xi = (spline_type == 4) ? kSogCubes2Xi4 : kSogCubes2Xi4;
  for (int k = 0; k < kSogCubes2NumNodes4; ++k) {
    sog_build_monomials_for_node(kSogCubes2Nodes4[k], xi, node_mono[k]);
  }

  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;
  const double dx_grid = mesh_lx / static_cast<double>(mesh_nx);
  const double dy_grid = mesh_ly / static_cast<double>(mesh_ny);
  const double dz_grid = mesh_lz / static_cast<double>(mesh_nz);

  // Precompute 1D integrals I_p(α) for each k-mode per axis.
  std::vector<std::complex<double>> Ipx[4];
  for (int p = 0; p < 4; ++p) {
    Ipx[p].resize(static_cast<size_t>(mesh_nx));
  }
  for (int ix = 0; ix < mesh_nx; ++ix) {
    const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
    const double alpha_x = twopi_over_x * static_cast<double>(kx_mode) * dx_grid;
    Ipx[0][static_cast<size_t>(ix)] = sog_I_int_0(alpha_x);
    Ipx[1][static_cast<size_t>(ix)] = sog_I_int_1(alpha_x);
    Ipx[2][static_cast<size_t>(ix)] = sog_I_int_2(alpha_x);
    Ipx[3][static_cast<size_t>(ix)] = sog_I_int_3(alpha_x);
  }
  std::vector<std::complex<double>> Ipy[4];
  for (int p = 0; p < 4; ++p) {
    Ipy[p].resize(static_cast<size_t>(mesh_ny));
  }
  for (int iy = 0; iy < mesh_ny; ++iy) {
    const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
    const double alpha_y = twopi_over_y * static_cast<double>(ky_mode) * dy_grid;
    Ipy[0][static_cast<size_t>(iy)] = sog_I_int_0(alpha_y);
    Ipy[1][static_cast<size_t>(iy)] = sog_I_int_1(alpha_y);
    Ipy[2][static_cast<size_t>(iy)] = sog_I_int_2(alpha_y);
    Ipy[3][static_cast<size_t>(iy)] = sog_I_int_3(alpha_y);
  }
  std::vector<std::complex<double>> Ipz[4];
  for (int p = 0; p < 4; ++p) {
    Ipz[p].resize(static_cast<size_t>(mesh_nz));
  }
  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double alpha_z = twopi_over_z * static_cast<double>(kz_mode) * dz_grid;
    Ipz[0][static_cast<size_t>(iz)] = sog_I_int_0(alpha_z);
    Ipz[1][static_cast<size_t>(iz)] = sog_I_int_1(alpha_z);
    Ipz[2][static_cast<size_t>(iz)] = sog_I_int_2(alpha_z);
    Ipz[3][static_cast<size_t>(iz)] = sog_I_int_3(alpha_z);
  }

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const double sqk = kx * kx + ky * ky + kz * kz;
        if (sqk == 0.0) continue;  // skip DC mode

        const size_t idx = mesh_index(ix, iy, iz);
        std::complex<double> phi_k(0.0, 0.0);

        for (int d = 0; d < kSogCubes2NumNodes4; ++d) {
          const auto &node = kSogCubes2Nodes4[d];
          const auto &mono = node_mono[d];

          const double phase = kx * static_cast<double>(node.dx) * dx_grid +
                               ky * static_cast<double>(node.dy) * dy_grid +
                               kz * static_cast<double>(node.dz) * dz_grid;
          const std::complex<double> eikd(std::cos(phase), std::sin(phase));

          std::complex<double> integral(0.0, 0.0);
          for (int m = 0; m < mono.num_terms; ++m) {
            const auto &term = mono.terms[m];
            const std::complex<double> prod =
                Ipx[term.px][static_cast<size_t>(ix)] *
                Ipy[term.py][static_cast<size_t>(iy)] *
                Ipz[term.pz][static_cast<size_t>(iz)];
            integral += term.coeff * prod;
          }
          phi_k += eikd * integral;
        }

        cubes2_influence_re[idx] = phi_k.real();
        cubes2_influence_im[idx] = phi_k.imag();
        const double abs_sq = phi_k.real() * phi_k.real() +
                              phi_k.imag() * phi_k.imag();
        cubes2_influence_sq[idx] = abs_sq;
      }
    }
  }

  delete[] node_mono;
}

void SOGKSpace::precompute_green_functions() {
  // Precompute per-k-point Green functions (geff_energy, geff, self_diag).
  // These depend on mesh geometry (box dimensions via k-vectors) and SOG
  // kernel parameters, but use precomputed sinc_pow tables (box-independent)
  // to avoid expensive sin/pow recomputation on every box change.
  //
  // The sinc_pow argument was originally:
  //   sinc_pow(0.5 * qx * dx, assign_pow)
  //   = sinc_pow(pi * (kx_mode/nx + jx), assign_pow)   // Lx cancels!
  // which depends only on grid indices, not on box dimensions.

  const size_t ngrid = static_cast<size_t>(mesh_nx) *
                       static_cast<size_t>(mesh_ny) *
                       static_cast<size_t>(mesh_nz);

  mesh_green_energy.assign(ngrid, 0.0);
  mesh_green_force.assign(ngrid, 0.0);
  mesh_green_self.assign(ngrid, 0.0);
  mesh_green_virial.assign(ngrid, 0.0);

  // k_sq_max: CubeS₂ uses grid Nyquist (all principal modes); B-spline uses
  // (2π/n_dl)² to bound the alias loop cost. The SOG kernel kfac = K(k²)
  // decays exponentially, so any modes beyond ～(2π/n_dl)² contribute
  // negligibly regardless — both cutoffs are numerically equivalent.
  double k_sq_max;
  if (spline_type >= 4) {
    const double dx = mesh_lx / static_cast<double>(mesh_nx);
    const double dy = mesh_ly / static_cast<double>(mesh_ny);
    const double dz = mesh_lz / static_cast<double>(mesh_nz);
    k_sq_max = MY_PI * MY_PI * (1.0 / (dx * dx) + 1.0 / (dy * dy) +
                                   1.0 / (dz * dz));
  } else {
    k_sq_max = (MY_2PI / n_dl) * (MY_2PI / n_dl);
  }

  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;
  const int alias_extent = mesh_alias_extent;
  const int alias_cnt = 2 * alias_extent + 1;

  // ── Fast-path check: can any alias possibly contribute? ──
  // For axis α, the alias spacing is Δα = 2π/Lα × n_mesh_α.
  // The smallest |q| for a j=±1 alias is at least |Δα - |k_α||,
  // and the minimum over all k-points is (Δα - k_max) where
  // k_max = sqrt(k_sq_max).  If (Δα - k_max)² > k_sq_max for all
  // axes, then NO alias from ANY axis can contribute for ANY k-point,
  // and we can skip the 3D alias loop entirely.
  const double k_max = std::sqrt(k_sq_max);
  const bool alias_fast_path =
      (twopi_over_x * static_cast<double>(mesh_nx) > 2.0 * k_max) &&
      (twopi_over_y * static_cast<double>(mesh_ny) > 2.0 * k_max) &&
      (twopi_over_z * static_cast<double>(mesh_nz) > 2.0 * k_max);

  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const double sqk = kx * kx + ky * ky + kz * kz;
        if (!(sqk > 0.0 && sqk <= k_sq_max)) {
          continue;
        }

        if (spline_type >= 4) {
          // CubeS₂ Green function. The grid is oversampled past the kernel
          // cutoff (mesh_oversample >= 1.0), so no alias can contribute — the
          // same regime as the B-spline alias fast-path — and the principal-mode
          // analytic influence |Φ(k)|² suffices: geff = K(k²) / |Φ(k)|².
          const size_t idx = mesh_index(ix, iy, iz);
          const double inf_sq = cubes2_influence_sq[idx];
          if (!(inf_sq > 1e-20) || !std::isfinite(inf_sq)) {
            continue;
          }
          const double kfac = spectral_kernel(sqk);
          const double vkern = virial_kernel(sqk);
          mesh_green_energy[idx] = kfac / inf_sq;
          mesh_green_force[idx] = kfac / inf_sq;  // alias fast-path
          mesh_green_self[idx] = kfac;
          mesh_green_virial[idx] = vkern / inf_sq;
          continue;
        }

        const double sz_sum = sinc_sum_z[static_cast<size_t>(iz)];
        const double sy_sum = sinc_sum_y[static_cast<size_t>(iy)];
        const double sx_sum = sinc_sum_x[static_cast<size_t>(ix)];
        const double denom_lin = sx_sum * sy_sum * sz_sum;
        const double denominator = denom_lin * denom_lin;
        if (!(denominator > 1e-20) || !std::isfinite(denominator)) {
          continue;
        }

        const size_t tbl_base_x =
            static_cast<size_t>(ix) * static_cast<size_t>(alias_cnt);
        const size_t tbl_base_y =
            static_cast<size_t>(iy) * static_cast<size_t>(alias_cnt);
        const size_t tbl_base_z =
            static_cast<size_t>(iz) * static_cast<size_t>(alias_cnt);

        double sum0, sum1;

        if (alias_fast_path) {
          // ── Fast path: only principal mode (j=0,0,0) contributes ──
          // No alias can lie within k_sq_max for any k-point, so
          // sum0 = K(k²) × w_x[0] × w_y[0] × w_z[0]
          // sum1 = k² × sum0   (principal mode: dot1 = kx²+ky²+kz² = k²)
          const double w2x = sinc_table_x[tbl_base_x +
                                          static_cast<size_t>(alias_extent)];
          const double w2y = sinc_table_y[tbl_base_y +
                                          static_cast<size_t>(alias_extent)];
          const double w2z = sinc_table_z[tbl_base_z +
                                          static_cast<size_t>(alias_extent)];
          const double w2_principal = w2x * w2y * w2z;

          const double kfac = spectral_kernel(sqk);
          const double vkern = virial_kernel(sqk);
          sum0 = kfac * w2_principal;
          sum1 = sqk * sum0;

          const size_t idx = mesh_index(ix, iy, iz);
          const double geff_energy = sum0 / denominator;
          // geff = sum1/(sqk*denom) = sum0/denom = geff_energy (fast path)

          mesh_green_energy[idx] = geff_energy;
          mesh_green_force[idx] = geff_energy;
          mesh_green_virial[idx] = vkern * w2_principal / denominator;
          if (w2_principal > 1e-20) {
            mesh_green_self[idx] = sum0 / w2_principal;
          }
        } else {
          // ── Full path: aliases may contribute ──
          sum0 = 0.0;
          sum1 = 0.0;
          double sum_virial = 0.0;
          for (int jx = -alias_extent; jx <= alias_extent; ++jx) {
            const double qx =
                twopi_over_x * static_cast<double>(kx_mode + mesh_nx * jx);
            const double wx_alias =
                sinc_table_x[tbl_base_x +
                             static_cast<size_t>(jx + alias_extent)];

            for (int jy = -alias_extent; jy <= alias_extent; ++jy) {
              const double qy =
                  twopi_over_y * static_cast<double>(ky_mode + mesh_ny * jy);
              const double wy_alias =
                  sinc_table_y[tbl_base_y +
                               static_cast<size_t>(jy + alias_extent)];

              for (int jz = -alias_extent; jz <= alias_extent; ++jz) {
                const double qz =
                    twopi_over_z * static_cast<double>(kz_mode + mesh_nz * jz);
                const double wz_alias =
                    sinc_table_z[tbl_base_z +
                                 static_cast<size_t>(jz + alias_extent)];

                const double qsq = qx * qx + qy * qy + qz * qz;
                if (!(qsq > 0.0 && qsq <= k_sq_max)) {
                  continue;
                }

                const double kfac_alias = spectral_kernel(qsq);
                if (!std::isfinite(kfac_alias) || kfac_alias == 0.0) {
                  continue;
                }

                const double wprod = wx_alias * wy_alias * wz_alias;
                sum0 += kfac_alias * wprod;
                const double dot1 = kx * qx + ky * qy + kz * qz;
                sum1 += dot1 * kfac_alias * wprod;

                // Virial: sum over aliases of K_v(q²) * wprod
                const double kfac_virial = virial_kernel(qsq);
                if (std::isfinite(kfac_virial) && kfac_virial != 0.0) {
                  sum_virial += kfac_virial * wprod;
                }
              }
            }
          }

          const size_t idx = mesh_index(ix, iy, iz);

          const double geff_energy = sum0 / denominator;
          const double geff = sum1 / (sqk * denominator);

          if (!std::isfinite(geff_energy) || !std::isfinite(geff)) {
            continue;
          }

          mesh_green_energy[idx] = geff_energy;
          mesh_green_force[idx] = geff;
          mesh_green_virial[idx] = sum_virial / denominator;

          // Self-interaction diag term: sum0 / W2(principal k)
          const double w2x = sinc_table_x[tbl_base_x +
                                          static_cast<size_t>(alias_extent)];
          const double w2y = sinc_table_y[tbl_base_y +
                                          static_cast<size_t>(alias_extent)];
          const double w2z = sinc_table_z[tbl_base_z +
                                          static_cast<size_t>(alias_extent)];
          const double w2_principal = w2x * w2y * w2z;
          if (w2_principal > 1e-20) {
            mesh_green_self[idx] = sum0 / w2_principal;
          }
        }
      }
    }
  }
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
    // RBSOG real-space self-energy correction (FINUFFT path)
    energy_local -= self_coeff * qsqsum_local;
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

    if (spline_type >= 4) {
      // CubeS₂ charge spreading.
      // spline_type == 6: 6th-order weight polynomials need verification
      // against paper Appendix Eq. A9. Falling back to 4th-order.
      const double xi = kSogCubes2Xi4;
      const double q_scaled = rho_scale * q[i];
      for (int k = 0; k < kSogCubes2NumNodes4; ++k) {
        const auto &node = kSogCubes2Nodes4[k];
        const double w = sog_cubes2_weight_4(tx, ty, tz, node, xi);
        if (w == 0.0) {
          continue;
        }
        const int igx = wrap_index(ix0 + node.dx, mesh_nx);
        const int igy = wrap_index(iy0 + node.dy, mesh_ny);
        const int igz = wrap_index(iz0 + node.dz, mesh_nz);
        const size_t idx = mesh_index(igx, igy, igz);
        mesh_rho[idx] += static_cast<FFT_SCALAR>(q_scaled * w);
      }
    } else {
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
  }

  const size_t ngrid = mesh_rho.size();
  for (size_t idx = 0; idx < ngrid; ++idx) {
    mesh_fft_work[2 * idx] = mesh_rho[idx];
    mesh_fft_work[2 * idx + 1] = 0.0;
  }

  mesh_fft->compute(mesh_fft_work.data(), mesh_fft_work.data(), FFT3d::FORWARD);

  const double scaleinv = 1.0 / static_cast<double>(ngrid);
  const double s2 = scaleinv * scaleinv;
  const double twopi_over_x = MY_2PI / mesh_lx;
  const double twopi_over_y = MY_2PI / mesh_ly;
  const double twopi_over_z = MY_2PI / mesh_lz;

  double energy_local = 0.0;
  double diag_sum_local = 0.0;
  std::array<double, 6> virial_local = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};     // analytic

  // Use precomputed Green functions (built once per mesh rebuild in
  // precompute_green_functions()).  This eliminates the per-step
  // 6-deep nested alias sum and spectral_kernel recomputation.
  for (int iz = 0; iz < mesh_nz; ++iz) {
    const int kz_mode = iz - mesh_nz * (2 * iz / mesh_nz);
    const double kz = twopi_over_z * static_cast<double>(kz_mode);

    for (int iy = 0; iy < mesh_ny; ++iy) {
      const int ky_mode = iy - mesh_ny * (2 * iy / mesh_ny);
      const double ky = twopi_over_y * static_cast<double>(ky_mode);

      for (int ix = 0; ix < mesh_nx; ++ix) {
        const int kx_mode = ix - mesh_nx * (2 * ix / mesh_nx);
        const double kx = twopi_over_x * static_cast<double>(kx_mode);

        const size_t idx = mesh_index(ix, iy, iz);

        const double geff_energy = mesh_green_energy[idx];
        const double geff = mesh_green_force[idx];

        // Zero gradients for k-points without a valid precomputed Green
        // function (sqk==0, sqk>k_sq_max, or non-finite).
        if (geff == 0.0 && geff_energy == 0.0) {
          mesh_gradx[2 * idx] = mesh_gradx[2 * idx + 1] = 0.0;
          mesh_grady[2 * idx] = mesh_grady[2 * idx + 1] = 0.0;
          mesh_gradz[2 * idx] = mesh_gradz[2 * idx + 1] = 0.0;
          continue;
        }

        // Accumulate self-interaction terms
        diag_sum_local += mesh_green_self[idx];

        const double rho_re = static_cast<double>(mesh_fft_work[2 * idx]);
        const double rho_im = static_cast<double>(mesh_fft_work[2 * idx + 1]);

        if (want_energy_global) {
          energy_local +=
              s2 * geff_energy * (rho_re * rho_re + rho_im * rho_im);
        }

        // ── Analytic k-space virial (rbsog-npt.md §3.1) ──
        // For orthogonal boxes this should match Σ r_i·F_i exactly.
        // Accumulated here as a debug cross-check against the force-based virial.
        //   W_{αβ}^F = s2 · Σ_k |ρ(k)|² · (G_energy·δ_{αβ} - G_virial·k_α·k_β)
        // where G_energy = K(k²)/|Φ(k)|²,  G_virial = virial_kernel(k²)/|Φ(k)|²
        if (want_virial_global) {
          const double rho_sq = rho_re * rho_re + rho_im * rho_im;
          const double geff_v = mesh_green_virial[idx];
          virial_local[0] += s2 * rho_sq * (geff_energy - geff_v * kx * kx);
          virial_local[1] += s2 * rho_sq * (geff_energy - geff_v * ky * ky);
          virial_local[2] += s2 * rho_sq * (geff_energy - geff_v * kz * kz);
          virial_local[3] += s2 * rho_sq * (-geff_v * kx * ky);
          virial_local[4] += s2 * rho_sq * (-geff_v * kx * kz);
          virial_local[5] += s2 * rho_sq * (-geff_v * ky * kz);
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
  std::array<double, 6> virial_local_rF = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // r_i·F_i

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

    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;

    if (spline_type >= 4) {
      // CubeS₂ force interpolation.
      // spline_type == 6 falls back to 4th-order (see spread comment).
      const double xi = kSogCubes2Xi4;
      for (int k = 0; k < kSogCubes2NumNodes4; ++k) {
        const auto &node = kSogCubes2Nodes4[k];
        const double w = sog_cubes2_weight_4(tx, ty, tz, node, xi);
        if (w == 0.0) {
          continue;
        }
        const int igx = wrap_index(ix0 + node.dx, mesh_nx);
        const int igy = wrap_index(iy0 + node.dy, mesh_ny);
        const int igz = wrap_index(iz0 + node.dz, mesh_nz);
        const size_t idx = mesh_index(igx, igy, igz);
        gx += w * static_cast<double>(mesh_gradx[2 * idx]);
        gy += w * static_cast<double>(mesh_grady[2 * idx]);
        gz += w * static_cast<double>(mesh_gradz[2 * idx]);
      }
    } else {
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
            const double w = wx[a] * wy[b] * wz[c];
            gx += w * static_cast<double>(mesh_gradx[2 * idx]);
            gy += w * static_cast<double>(mesh_grady[2 * idx]);
            gz += w * static_cast<double>(mesh_gradz[2 * idx]);
          }
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
      virial_local_rF[0] += x[i][0] * fxs;
      virial_local_rF[1] += x[i][1] * fys;
      virial_local_rF[2] += x[i][2] * fzs;
      virial_local_rF[3] += x[i][0] * fys;
      virial_local_rF[4] += x[i][0] * fzs;
      virial_local_rF[5] += x[i][1] * fzs;
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
    // RBSOG real-space self-energy correction
    energy -= self_coeff * qsqsum_all;
    energy *= qscale_local;
  }

  if (want_virial_global) {
    double virial_all[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double virial_rF_all[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(virial_local.data(), virial_all, 6, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(virial_local_rF.data(), virial_rF_all, 6, MPI_DOUBLE, MPI_SUM, world);

    // Analytic virial as default (rbsog-npt.md §3.1):
    //   W_{αβ} = 0.5·V·qscale · s2 · Σ_k |ρ|² (G_energy·δ_{αβ} - G_virial·k_α·k_β)
    const double virial_scale = 0.5 * volume_local * qscale_local;
    for (int j = 0; j < 6; ++j) {
      virial[j] = virial_scale * virial_all[j];
    }

    // Debug: compare analytic vs Σ r_i·F_i on first mesh build
    if (comm->me == 0) {
      std::string ss = "SOG virial check (analytic vs r_i-F_i):";
      const char *names[6] = {"xx","yy","zz","xy","xz","yz"};
      for (int j = 0; j < 6; ++j) {
        double ana = virial_scale * virial_all[j];
        double delta = ana - virial_rF_all[j];
        double rel = (std::abs(virial_rF_all[j]) > 1e-10)
                         ? std::abs(delta / virial_rF_all[j]) * 100.0
                         : 0.0;
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      " %s: ana=%.6g rF=%.6g diff=%.3g (%.2f%%)",
                      names[j], ana, virial_rF_all[j], delta, rel);
        ss += buf;
      }
      utils::logmesg(lmp, ss + "\n");
    }
  }
}

void SOGKSpace::compute(int eflag, int vflag) {
  ev_init(eflag, vflag, 0);

  // atom->q can be updated every step by pair_style deepmd
  // (latent_charge_to_q), so we must refresh charge moments each compute.
  qsum_qsq();
  natoms_original = atom->natoms;

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
      mesh_gradz.capacity() * sizeof(FFT_SCALAR) +
      (cubes2_influence_re.capacity() + cubes2_influence_im.capacity() +
       cubes2_influence_sq.capacity()) *
          sizeof(double);
  const size_t param_bytes =
      (amp.capacity() + bandwidth.capacity()) * sizeof(double);
  return static_cast<double>(mesh_bytes + param_bytes);
}
