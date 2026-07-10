// SPDX-License-Identifier: LGPL-3.0-or-later
#include "fix_sog_response.h"

#include "atom.h"
#include "comm.h"
#include <sstream>
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixSOGResponse::FixSOGResponse(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg),
      pair_deepmd(nullptr),
      sog_kspace(nullptr),
      nchannels(1),
      pair_deepmd_index(0) {
  if (narg < 4)
    error->all(FLERR,
               "Illegal fix sog/response command: need at least 'model' keyword");
  ener_unit_cvt_factor = 1.0;
  dist_unit_cvt_factor = 1.0;
  force_unit_cvt_factor = 1.0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix sog/response missing model path");
      model = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "pair_deepmd_index") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix sog/response missing pair_deepmd_index value");
      pair_deepmd_index = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "fused") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix sog/response missing fused value");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        fused_mode = true;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        fused_mode = false;
      else
        error->all(FLERR, "fix sog/response fused expects yes or no");
      iarg += 2;
    } else {
      error->all(FLERR, fmt::format("Unknown fix sog/response keyword: {}", arg[iarg]));
    }
  }
  if (model.empty()) error->all(FLERR, "fix sog/response: must specify 'model'");

  // The second DeepPot `dp` is ONLY used by the non-fused fallback (the post_force catch branch).
  // In fused mode that branch is never reached, so skip this redundant, GPU-resident model load.
  if (!fused_mode) dp.init(model, 0);

  // locate the deepmd pair
  int ix{0};
  pair_deepmd = dynamic_cast<PairDeepMD*>(
      force->pair_match("deepmd", 1, pair_deepmd_index > 0 ? pair_deepmd_index : -1));
  if (!pair_deepmd)
    error->all(FLERR, "fix sog/response requires pair_style deepmd");

  // Fuse: have the pair retain its forward autograd graph so the charge-response
  // correction is a VJP-only backward here, not a redundant second model forward.
  // fused_mode goes further: the pair runs an energy-only+charge forward (no extended_force),
  // and this fix does ONE combined backward for the total short-range+response force.
  if (fused_mode)
    pair_deepmd->enable_charge_only_forward();
  else
    pair_deepmd->enable_charge_response_graph();

  // deepmd unit conversions from the pair
  ener_unit_cvt_factor = pair_deepmd->ener_unit_cvt_factor;
  dist_unit_cvt_factor = pair_deepmd->dist_unit_cvt_factor;
  force_unit_cvt_factor = pair_deepmd->force_unit_cvt_factor;

  nchannels = pair_deepmd->ncharge_channels;
  if (nchannels < 1) nchannels = 1;

  // build type_idx_map (clone fix_dplr:160-200 pattern)
  {
    std::vector<std::string> type_names = pair_deepmd->type_names;
    std::vector<std::string> type_map_vec;
    std::string type_map_str;
    if (!fused_mode) dp.get_type_map(type_map_str);  // dp is uninitialized in fused mode; type_idx_map is unused there
    std::istringstream iss(type_map_str);
    std::string tn;
    while (iss >> tn) type_map_vec.push_back(tn);
    if (type_names.empty() || type_map_vec.empty()) {
      type_idx_map.resize(atom->ntypes);
      for (int ii = 0; ii < atom->ntypes; ++ii) type_idx_map[ii] = ii;
    } else {
      for (auto &tn_name : type_names) {
        bool found = false;
        for (size_t jj = 0; jj < type_map_vec.size(); ++jj) {
          if (type_map_vec[jj] == tn_name) {
            type_idx_map.push_back(static_cast<int>(jj));
            found = true;
            break;
          }
        }
        if (!found) type_idx_map.push_back(-1);
      }
    }
  }

  comm_reverse = 3;  // dfcorr_buff: 3 doubles per atom
  virial_global_flag = 1;
  energy_global_flag = 1;
}

FixSOGResponse::~FixSOGResponse() {}

int FixSOGResponse::setmask() {
  int mask = 0;
  mask |= FixConst::PRE_FORCE;
  mask |= FixConst::POST_FORCE;
  mask |= FixConst::MIN_PRE_FORCE;
  mask |= FixConst::MIN_POST_FORCE;
  // (no THERMO_ENERGY in the fix mask — energy tally is handled by the pair+kspace)
  return mask;
}

void FixSOGResponse::init() {
  // locate the SOG kspace solver
  sog_kspace = dynamic_cast<SOGKSpace*>(force->kspace_match("sog", 1));
  if (!sog_kspace)
    error->all(FLERR,
               "fix sog/response requires kspace_style sog (not fastsog/pppm/...): "
               "the SOG kspace solver must be active to provide v_i via get_potential().");
}

void FixSOGResponse::setup(int vflag) { post_force(vflag); }
void FixSOGResponse::min_setup(int vflag) { setup(vflag); }
void FixSOGResponse::setup_pre_force(int vflag) { pre_force(vflag); }
void FixSOGResponse::min_pre_force(int vflag) { pre_force(vflag); }
void FixSOGResponse::min_post_force(int vflag) { post_force(vflag); }

void FixSOGResponse::pre_force(int /*vflag*/) {
  // Enable the SOG kspace solver to compute v_i = ∂E_k/∂q_i.
  // The Verlet force loop calls kspace->compute() between pre_force and
  // post_force, so vpot will be populated when post_force reads it.
  if (sog_kspace) sog_kspace->set_want_potential(true);
}

void FixSOGResponse::post_force(int vflag) {
  // Modified template from FixDPLR::post_force
  if (vflag) {
    v_setup(vflag);
  } else {
    evflag = 0;
  }
  // reject per-atom virial: global only
  if (vflag_atom) {
    error->all(FLERR, "fix sog/response does not support per-atom virial");
  }

  int nlocal = atom->nlocal;
  if (nlocal <= 0) return;
  int nall = nlocal + atom->nghost;

  // ── Read kspace potential v_i (needed by BOTH the cached VJP and the fallback) ──
  const std::vector<double> &vpot = sog_kspace->get_potential();
  if (static_cast<int>(vpot.size()) != nlocal)
    error->all(FLERR, "sog/response: vpot size != nlocal (forgot set_want_potential?)");

  // Build v_per_atom [nlocal * nchannels] (v_i replicated per channel)
  std::vector<double> v_per_atom(nlocal * nchannels, 0.0);
  for (int i = 0; i < nlocal; ++i) {
    double vi = vpot[i] / ener_unit_cvt_factor;
    for (int c = 0; c < nchannels; ++c)
      v_per_atom[i * nchannels + c] = vi;
  }

  // ── Compute the charge-response correction ──
  // Fast path: VJP on the pair's retained forward graph (no second forward, no model inputs).
  // Fallback: standalone forward+backward — its dcoord/dtype/dbox packing and the message-passing
  // InputNlist + mapping are built ONLY here (deferred off the hot path; the cached path never
  // reads them, so on the steady-state fast path this O(nall) per-step work is skipped entirely).
  std::vector<double> dfcorr, dvcorr;
  bool used_cached = false;
  if (fused_mode) {
    // Tier-2: the pair ran the energy-only+charge forward (added zero force). This ONE combined
    // backward yields the TOTAL short-range+response force (+ total virial). No fallback: a missing
    // retained graph would silently drop the short-range force, so surface a hard error instead.
    pair_deepmd->compute_combined_response_cached(dfcorr, dvcorr, v_per_atom);
    used_cached = true;
  } else {
  try {
    pair_deepmd->compute_charge_response_cached(dfcorr, dvcorr, v_per_atom);
    used_cached = true;
  } catch (deepmd_compat::deepmd_exception &) {
    double **x = atom->x;
    int *type = atom->type;
    std::vector<double> dcoord(nall * 3, 0.0);
    std::vector<int> dtype(nall);
    for (int ii = 0; ii < nall; ++ii) {
      dtype[ii] = type[ii] - 1;
      dcoord[ii * 3 + 0] = (x[ii][0] * dist_unit_cvt_factor + domain->boxlo[0]);
      dcoord[ii * 3 + 1] = (x[ii][1] * dist_unit_cvt_factor + domain->boxlo[1]);
      dcoord[ii * 3 + 2] = (x[ii][2] * dist_unit_cvt_factor + domain->boxlo[2]);
    }
    std::vector<double> dbox(9, 0.0);
    dbox[0] = (domain->boxhi[0] - domain->boxlo[0]) * dist_unit_cvt_factor;
    dbox[4] = (domain->boxhi[1] - domain->boxlo[1]) * dist_unit_cvt_factor;
    dbox[8] = (domain->boxhi[2] - domain->boxlo[2]) * dist_unit_cvt_factor;
    dbox[3] = domain->xy * dist_unit_cvt_factor;
    dbox[6] = domain->xz * dist_unit_cvt_factor;
    dbox[7] = domain->yz * dist_unit_cvt_factor;
    NeighList *list = pair_deepmd->list;
    int nghost_val = nall - nlocal;
    CommBrickDeepMD *commdata = (CommBrickDeepMD *)comm;
    deepmd::InputNlist lmp_list(
        list->inum, list->ilist, list->numneigh, list->firstneigh,
        commdata->nswap, commdata->sendnum, commdata->recvnum,
        commdata->firstrecv, commdata->sendlist, commdata->sendproc,
        commdata->recvproc, &world);
    lmp_list.set_mask(NEIGHMASK);
    std::vector<int> mapping_vec(nall, -1);
    if (comm->nprocs == 1 && atom->map_style != Atom::MAP_NONE) {
      for (int ii = 0; ii < nall; ++ii)
        mapping_vec[ii] = atom->map(atom->tag[ii]);
      lmp_list.set_mapping(mapping_vec.data());
    }
    dp.compute_charge_response(dfcorr, dvcorr, v_per_atom, dcoord, dtype, dbox,
                               nghost_val, lmp_list, 0);
  }
  }
  static bool warned_fallback = false;
  if (!used_cached && !warned_fallback && comm->me == 0) {
    warned_fallback = true;
    error->warning(FLERR,
                   "fix sog/response: charge-response graph unavailable; using "
                   "slower standalone forward (fusion inactive).");
  }

  // Convert correction back to LAMMPS units
  for (size_t ii = 0; ii < dfcorr.size(); ++ii)
    dfcorr[ii] *= force_unit_cvt_factor;
  for (int ii = 0; ii < 9; ++ii)
    dvcorr[ii] *= ener_unit_cvt_factor;

  // ── Reverse-comm the ghost-atom correction forces ──
  dfcorr_buff = dfcorr;
#if LAMMPS_VERSION_NUMBER >= 20220324
  comm->reverse_comm(this, 3);
#else
  comm->reverse_comm_fix(this, 3);
#endif
  std::copy(dfcorr_buff.begin(), dfcorr_buff.end(), dfcorr.begin());

  // ── Accumulate into atom->f ──
  double **f = atom->f;
  for (int ii = 0; ii < nlocal; ++ii) {
    f[ii][0] += dfcorr[ii * 3 + 0];
    f[ii][1] += dfcorr[ii * 3 + 1];
    f[ii][2] += dfcorr[ii * 3 + 2];
  }

  // ── Global virial (no DPLR extra outer-product term — the model's
  //    charge_response_virial already contains F⊗r) ──
  if (evflag) {
    double vv[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    vv[0] += dvcorr[0];
    vv[1] += dvcorr[4];
    vv[2] += dvcorr[8];
    vv[3] += dvcorr[3];
    vv[4] += dvcorr[6];
    vv[5] += dvcorr[7];
    v_tally(0, vv);
  }
}

int FixSOGResponse::pack_reverse_comm(int n, int first, double *buf) {
  int offset = first * 3;
  int last = offset + n * 3;
  for (int i = offset; i < last; ++i) { buf[i - offset] = dfcorr_buff[i]; }
  return 3;
}

void FixSOGResponse::unpack_reverse_comm(int n, int *list, double *buf) {
  for (int ii = 0; ii < n; ++ii) {
    int idx = list[ii] * 3;
    dfcorr_buff[idx + 0] += buf[ii * 3 + 0];
    dfcorr_buff[idx + 1] += buf[ii * 3 + 1];
    dfcorr_buff[idx + 2] += buf[ii * 3 + 2];
  }
}
