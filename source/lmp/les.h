// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(les, LESKSpace)
// clang-format on
#else

#ifndef LMP_LES_H
#define LMP_LES_H

#include <string>

#include "pppm.h"

namespace LAMMPS_NS {

class LESKSpace : public PPPM {
 public:
  explicit LESKSpace(class LAMMPS* lmp);
  ~LESKSpace() override;

  void settings(int narg, char** arg) override;
  void init() override;

 private:
  bool parse_bool_token(const std::string& token, bool& value) const;
  void finalize_options() const;
  void apply_sigma_mapping();
  double sigma_to_gewald(const double sigma) const;

  double accuracy_in;
  double n_dl;
  bool remove_self_interaction;
  double sigma_param;
};

}  // namespace LAMMPS_NS

#endif
#endif