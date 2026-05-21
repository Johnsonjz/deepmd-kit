// SPDX-License-Identifier: LGPL-3.0-or-later
// Test DeepPot::compute_with_charge on PT backend.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeepPot.h"
#include "neighbor_list.h"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepPotChargePt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  deepmd::DeepPot dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init("../../tests/infer/deeppot_sea.pth");
  }

  void TearDown() override {}
};

TYPED_TEST_SUITE(TestInferDeepPotChargePt, ValueTypes);

TYPED_TEST(TestInferDeepPotChargePt, cpu_lmp_nlist_charge_optional_output) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  deepmd::DeepPot& dp = this->dp;

  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);

  double ener_ref = 0.0;
  std::vector<VALUETYPE> force_ref_raw, force_ref, virial_ref;
  dp.compute(ener_ref, force_ref_raw, virial_ref, coord_cpy, atype_cpy, box,
             nall - nloc, inlist, 0);
  _fold_back<VALUETYPE>(force_ref, force_ref_raw, mapping, nloc, nall, 3);

  double ener_new = 0.0;
  std::vector<VALUETYPE> force_new_raw, force_new, virial_new;
  std::vector<VALUETYPE> atom_energy, atom_virial, atom_charge;
  dp.compute_with_charge(ener_new, force_new_raw, virial_new, atom_energy,
                         atom_virial, atom_charge, coord_cpy, atype_cpy, box,
                         nall - nloc, inlist, 0);
  _fold_back<VALUETYPE>(force_new, force_new_raw, mapping, nloc, nall, 3);

  EXPECT_LT(fabs(ener_new - ener_ref), EPSILON);
  ASSERT_EQ(force_new.size(), force_ref.size());
  ASSERT_EQ(virial_new.size(), virial_ref.size());
  for (size_t ii = 0; ii < force_new.size(); ++ii) {
    EXPECT_LT(fabs(force_new[ii] - force_ref[ii]), EPSILON);
  }
  for (size_t ii = 0; ii < virial_new.size(); ++ii) {
    EXPECT_LT(fabs(virial_new[ii] - virial_ref[ii]), EPSILON);
  }

  // Non-SOG models do not expose latent_charge; API returns empty charge list.
  EXPECT_TRUE(atom_charge.empty());
}
