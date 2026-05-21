// SPDX-License-Identifier: LGPL-3.0-or-later

#include "les.h"

#include <math.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

#include "error.h"

using namespace LAMMPS_NS;

namespace {

constexpr double kLESDefaultSigma = 1.0;

std::string to_lower_copy(const std::string& in) {
  std::string out = in;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

}  // namespace

LESKSpace::LESKSpace(LAMMPS* lmp)
    : PPPM(lmp),
      accuracy_in(1e-6),
      n_dl(1.0),
      remove_self_interaction(true),
      sigma_param(kLESDefaultSigma) {
  triclinic_support = 1;
}

LESKSpace::~LESKSpace() {}

bool LESKSpace::parse_bool_token(const std::string& token, bool& value) const {
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

double LESKSpace::sigma_to_gewald(const double sigma) const {
  return 1.0 / (std::sqrt(2.0) * sigma);
}

void LESKSpace::finalize_options() const {
  if (!(std::isfinite(accuracy_in) && accuracy_in > 0.0)) {
    error->all(FLERR, "kspace style les requires a positive accuracy argument");
  }
  if (!(std::isfinite(n_dl) && n_dl > 0.0)) {
    error->all(FLERR, "kspace style les requires n_dl > 0");
  }
  if (!(std::isfinite(sigma_param) && sigma_param > 0.0)) {
    error->all(FLERR, "kspace style les requires sigma > 0");
  }

  // LES long-range splitting is defined to be Ewald-compatible with PPPM.
  // We currently require RSI on to stay consistent with PPPM's default self-term.
  if (!remove_self_interaction) {
    error->all(FLERR,
               "kspace style les currently requires remove_self_interaction yes");
  }
}

void LESKSpace::apply_sigma_mapping() {
  const double mapped_gewald = sigma_to_gewald(sigma_param);
  if (!(std::isfinite(mapped_gewald) && mapped_gewald > 0.0)) {
    error->all(FLERR, "kspace style les produced invalid g_ewald from sigma");
  }

  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss << std::setprecision(17) << mapped_gewald;
  std::string gewald_str = oss.str();

  char* modarg[2];
  modarg[0] = const_cast<char*>("gewald");
  modarg[1] = const_cast<char*>(gewald_str.c_str());
  KSpace::modify_params(2, modarg);
}

void LESKSpace::settings(int narg, char** arg) {
  if (narg < 1) {
    error->all(FLERR, "Illegal kspace_style les command");
  }

  accuracy_in = std::fabs(atof(arg[0]));

  char* base_arg[1];
  base_arg[0] = arg[0];
  PPPM::settings(1, base_arg);

  n_dl = 1.0;
  remove_self_interaction = true;
  sigma_param = kLESDefaultSigma;

  int iarg = 1;
  while (iarg < narg) {
    const std::string key = to_lower_copy(arg[iarg]);
    if (key == "n_dl") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style les missing n_dl value");
      }
      n_dl = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (key == "remove_self_interaction") {
      if (iarg + 1 >= narg) {
        error->all(FLERR,
                   "kspace style les missing remove_self_interaction value");
      }
      bool val = false;
      if (!parse_bool_token(arg[iarg + 1], val)) {
        error->all(
            FLERR,
            "kspace style les remove_self_interaction expects yes/no token");
      }
      remove_self_interaction = val;
      iarg += 2;
    } else if (key == "sigma") {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "kspace style les missing sigma value");
      }
      sigma_param = atof(arg[iarg + 1]);
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal kspace_style les command");
    }
  }

  finalize_options();
}

void LESKSpace::init() {
  if (differentiation_flag != 0) {
    error->all(FLERR,
               "kspace style les currently supports only kspace_modify diff ik");
  }

  finalize_options();
  apply_sigma_mapping();
  PPPM::init();
}
