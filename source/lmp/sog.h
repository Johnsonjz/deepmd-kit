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

#include "kspace.h"
#include "lmpfftsettings.h"

namespace LAMMPS_NS {

class FFT3d;

class SOGKSpace : public KSpace {
 public:
  explicit SOGKSpace(class LAMMPS* lmp);
  ~SOGKSpace() override;

  void settings(int narg, char** arg) override;
  void init() override;
  void setup() override;
  void compute(int eflag, int vflag) override;
  double memory_usage() override;

 private:
  bool is_keyword(const std::string& token) const;
  bool parse_bool_token(const std::string& token, bool& value) const;
  void finalize_kernel_parameters();
  double spectral_kernel(const double sqk) const;
  bool compute_finufft(int eflag, int vflag);
  void compute_mesh_fft(int eflag, int vflag);

  void ensure_fft_plan();
  void destroy_fft_plan();

  size_t mesh_index(int ix, int iy, int iz) const;
  double periodic_fraction(double x, double xlo, double prd) const;

  double accuracy_in;
  double n_dl;
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
  bool mesh_ready;

  int mesh_nx;
  int mesh_ny;
  int mesh_nz;

  double mesh_lx;
  double mesh_ly;
  double mesh_lz;

  FFT3d* mesh_fft;

  std::vector<FFT_SCALAR> mesh_rho;
  std::vector<FFT_SCALAR> mesh_fft_work;
  std::vector<FFT_SCALAR> mesh_gradx;
  std::vector<FFT_SCALAR> mesh_grady;
  std::vector<FFT_SCALAR> mesh_gradz;

  std::vector<double> amp;
  std::vector<double> bandwidth;
};

}  // namespace LAMMPS_NS

#endif
#endif