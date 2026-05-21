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
  void compute_gf_ik() override;
  void fieldforce_ik() override;

 private:
  bool is_keyword(const std::string& token) const;
  bool parse_bool_token(const std::string& token, bool& value) const;
  void finalize_kernel_parameters();
  double kernel_prefactor(const double sqk) const;
  void rebuild_sog_greensfn();
  bool compute_finufft(int eflag, int vflag);

  double accuracy_in;
  double n_dl;
  bool remove_self_interaction;
  bool use_finufft;
  double finufft_eps;
  std::string finufft_library;
  bool finufft_warned;

  double b_param;
  double sigma_param;
  int m_param;
  double self_diag_sum;

  bool kernel_ready;
  std::vector<double> amp;
  std::vector<double> bandwidth;
  std::vector<double> fele;
};

}  // namespace LAMMPS_NS

#endif
#endif