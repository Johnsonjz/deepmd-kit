// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(sog, SOGKSpace)
// clang-format on
#else

#ifndef LMP_SOG_H
#define LMP_SOG_H

#include <string>
#include <vector>

#include "pppm.h"
#include "lmpfftsettings.h"

namespace LAMMPS_NS {

class SOGKSpace : public PPPM {
 public:
  explicit SOGKSpace(class LAMMPS* lmp);
  ~SOGKSpace() override;

  void settings(int narg, char** arg) override;
  void init() override;
  void setup() override;
  void compute(int eflag, int vflag) override;
  double memory_usage() override;

 protected:
  void set_grid_global() override;
  void compute_gf_ik() override;
  void poisson_ik() override;

 private:
  bool is_keyword(const std::string& token) const;
  bool parse_bool_token(const std::string& token, bool& value) const;
  bool try_import_n_dl_from_pair_model(bool strict_missing);
  void finalize_kernel_parameters();
  double spectral_kernel(const double sqk) const;

  double accuracy_in;
  double n_dl;
  bool n_dl_user_specified;
  bool n_dl_from_model;
  bool remove_self_interaction;
  bool use_finufft;
  double finufft_eps;
  std::string finufft_library;
  bool finufft_warned;

  double mesh_oversample;
  int mesh_alias_extent;

  double b_param;
  double sigma_param;
  int m_param;
  double self_diag_sum;

  bool kernel_ready;

  std::vector<double> greensfn_energy;

  std::vector<double> amp;
  std::vector<double> bandwidth;
};

}  // namespace LAMMPS_NS

#endif
#endif