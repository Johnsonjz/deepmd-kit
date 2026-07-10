// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef FIX_CLASS

FixStyle(sog/response, FixSOGResponse)

#else

#ifndef LMP_FIX_SOG_RESPONSE_H
#define LMP_FIX_SOG_RESPONSE_H

#include <stdio.h>

#include <map>

#include "fix.h"
#include "pair_deepmd.h"
#include "sog.h"
#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepPot.h"
#else
#include "deepmd/DeepPot.h"
#endif
namespace deepmd_compat = deepmd;
#else
#ifdef LMPPLUGIN
#include "deepmd.hpp"
#else
#include "deepmd/deepmd.hpp"
#endif
namespace deepmd_compat = deepmd::hpp;
#endif

namespace LAMMPS_NS {
class FixSOGResponse : public Fix {
 public:
  FixSOGResponse(class LAMMPS*, int, char**);
  ~FixSOGResponse() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void min_setup(int) override;
  void pre_force(int) override;
  void post_force(int) override;
  void min_pre_force(int) override;
  void min_post_force(int) override;
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;
  double ener_unit_cvt_factor, dist_unit_cvt_factor, force_unit_cvt_factor;

 private:
  PairDeepMD* pair_deepmd;
  deepmd_compat::DeepPot dp;
  SOGKSpace* sog_kspace;
  std::string model;
  std::vector<int> type_idx_map;
  std::vector<double> dfcorr_buff;
  int nchannels;
  int pair_deepmd_index;
  bool fused_mode = false;  // Tier-2: pair does energy-only forward, fix does one combined backward
};
}  // namespace LAMMPS_NS

#endif  // LMP_FIX_SOG_RESPONSE_H
#endif  // FIX_CLASS
