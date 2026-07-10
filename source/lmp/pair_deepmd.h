// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS

PairStyle(deepmd, PairDeepMD)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

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

#include <fstream>
#include <iostream>
#include <map>

#include "comm_brick.h"
#include "pair_base.h"
#define FLOAT_PREC double

namespace LAMMPS_NS {
class FixSOGResponse;
class CommBrickDeepMD : public CommBrick {
  friend class PairDeepMD;
  friend class FixSOGResponse;
};
class PairDeepMD : public PairDeepBaseModel {
 public:
  PairDeepMD(class LAMMPS*);
  ~PairDeepMD() override;
  void settings(int, char**) override;
  void coeff(int, char**) override;
  void compute(int, int) override;
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;

 protected:
  deepmd_compat::DeepPot deep_pot;
  deepmd_compat::DeepPotModelDevi deep_pot_model_devi;

 private:
  CommBrickDeepMD* commdata_;
  bool latent_charge_to_q;

 public:
  // Multi-channel charge access for kspace (fast_dp_sog)
  int ncharge_channels;
  std::vector<double> dcharge_multi;
#ifdef DP_USE_CXX_API
  // ── fix sog/response fusion: reuse this pair's retained forward graph ──
  void enable_charge_response_graph() {
    deep_pot.set_retain_charge_graph(true);
  }
  void compute_charge_response_cached(std::vector<double>& force_corr,
                                      std::vector<double>& virial_corr,
                                      const std::vector<double>& v_per_atom) {
    deep_pot.compute_charge_response_cached(force_corr, virial_corr, v_per_atom);
  }
  // ── Tier-2 fused: energy-only+charge forward here, one combined backward in the fix ──
  void enable_charge_only_forward() {
    deep_pot.set_retain_charge_graph(true);
    deep_pot.set_charge_only_forward(true);
  }
  void compute_combined_response_cached(std::vector<double>& force_total,
                                        std::vector<double>& virial_total,
                                        const std::vector<double>& v_per_atom) {
    deep_pot.compute_combined_response(force_total, virial_total, v_per_atom);
  }
#endif
};

}  // namespace LAMMPS_NS

#endif
#endif
