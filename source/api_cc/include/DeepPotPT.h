// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include "DeepPot.h"

namespace deepmd {
/**
 * @brief PyTorch implementation for Deep Potential.
 **/
class DeepPotPT : public DeepPotBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPotPT();
  virtual ~DeepPotPT();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepPotPT(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

 private:
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute the atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute the atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);

  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_with_charge(ENERGYVTYPE& ener,
                           std::vector<VALUETYPE>& force,
                           std::vector<VALUETYPE>& virial,
                           std::vector<VALUETYPE>& atom_energy,
                           std::vector<VALUETYPE>& atom_virial,
                           std::vector<VALUETYPE>& atom_charge,
                           const std::vector<VALUETYPE>& coord,
                           const std::vector<int>& atype,
                           const std::vector<VALUETYPE>& box,
                           const int nghost,
                           const InputNlist& lmp_list,
                           const int& ago,
                           const std::vector<VALUETYPE>& fparam,
                           const std::vector<VALUETYPE>& aparam,
                           const bool atomic);
  /**
   * @brief Charge-response correction force and virial.
   * @param[out] force_corr  Correction force, size nall*3.
   * @param[out] virial_corr Correction virial, size 9.
   * @param[in]  v_per_atom  Per-atom potential v_i, size nloc*nchannels.
   */
  template <typename VALUETYPE>
  void compute_charge_response(
      std::vector<VALUETYPE> &force_corr,
      std::vector<VALUETYPE> &virial_corr,
      const std::vector<VALUETYPE> &v_per_atom,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const int nghost,
      const InputNlist &lmp_list,
      const int &ago,
      const std::vector<VALUETYPE> &fparam,
      const std::vector<VALUETYPE> &aparam);
  /**
   * @brief Evaluate the energy, force, and virial with the mixed type
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The array should be of size nframes x
   *natoms.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute the atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(ENERGYVTYPE& ener,
                          std::vector<VALUETYPE>& force,
                          std::vector<VALUETYPE>& virial,
                          const int& nframes,
                          const std::vector<VALUETYPE>& coord,
                          const std::vector<int>& atype,
                          const std::vector<VALUETYPE>& box,
                          const std::vector<VALUETYPE>& fparam,
                          const std::vector<VALUETYPE>& aparam,
                          const bool atomic);
  /**
   * @brief Evaluate the energy, force, and virial with the mixed type
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The array should be of size nframes x
   *natoms.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Whether to compute the atomic energy and virial.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(ENERGYVTYPE& ener,
                          std::vector<VALUETYPE>& force,
                          std::vector<VALUETYPE>& virial,
                          std::vector<VALUETYPE>& atom_energy,
                          std::vector<VALUETYPE>& atom_virial,
                          const int& nframes,
                          const std::vector<VALUETYPE>& coord,
                          const std::vector<int>& atype,
                          const std::vector<VALUETYPE>& box,
                          const std::vector<VALUETYPE>& fparam,
                          const std::vector<VALUETYPE>& aparam,
                          const bool atomic);

 public:
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  /**
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const {
    assert(inited);
    return ntypes_spin;
  };
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const {
    assert(inited);
    return dfparam;
  };
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const {
    assert(inited);
    return daparam;
  };
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string& type_map);

  /**
   * @brief Get whether the atom dimension of aparam is nall instead of fparam.
   * @param[out] aparam_nall whether the atom dimension of aparam is nall
   *instead of fparam.
   **/
  bool is_aparam_nall() const {
    assert(inited);
    return aparam_nall;
  };
  /**
   * @brief Check if the model has default frame parameters.
   * @return true if the model has default frame parameters.
   **/
  bool has_default_fparam() const {
    assert(inited);
    return has_default_fparam_;
  };

  // forward to template class
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const int& ago,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);

  void computew_with_charge(std::vector<double>& ener,
                            std::vector<double>& force,
                            std::vector<double>& virial,
                            std::vector<double>& atom_energy,
                            std::vector<double>& atom_virial,
                            std::vector<double>& atom_charge,
                            const std::vector<double>& coord,
                            const std::vector<int>& atype,
                            const std::vector<double>& box,
                            const int nghost,
                            const InputNlist& inlist,
                            const int& ago,
                            const std::vector<double>& fparam,
                            const std::vector<double>& aparam,
                            const bool atomic);
  void computew_with_charge(std::vector<double>& ener,
                            std::vector<float>& force,
                            std::vector<float>& virial,
                            std::vector<float>& atom_energy,
                            std::vector<float>& atom_virial,
                            std::vector<float>& atom_charge,
                            const std::vector<float>& coord,
                            const std::vector<int>& atype,
                            const std::vector<float>& box,
                            const int nghost,
                            const InputNlist& inlist,
                            const int& ago,
                            const std::vector<float>& fparam,
                            const std::vector<float>& aparam,
                            const bool atomic);
  void computew_charge_response(
      std::vector<double> &force_corr, std::vector<double> &virial_corr,
      const std::vector<double> &v_per_atom,
      const std::vector<double> &coord, const std::vector<int> &atype,
      const std::vector<double> &box, const int nghost,
      const InputNlist &inlist, const int &ago,
      const std::vector<double> &fparam = {},
      const std::vector<double> &aparam = {});
  void computew_charge_response(
      std::vector<float> &force_corr, std::vector<float> &virial_corr,
      const std::vector<float> &v_per_atom,
      const std::vector<float> &coord, const std::vector<int> &atype,
      const std::vector<float> &box, const int nghost,
      const InputNlist &inlist, const int &ago,
      const std::vector<float> &fparam = {},
      const std::vector<float> &aparam = {});
  // ── Charge-response fusion: reuse the pair's retained forward graph ──
  void set_retain_charge_graph(bool b) { retain_charge_graph_ = b; }
  template <typename VALUETYPE>
  void compute_charge_response_cached(
      std::vector<VALUETYPE> &force_corr,
      std::vector<VALUETYPE> &virial_corr,
      const std::vector<VALUETYPE> &v_per_atom);
  void computew_charge_response_cached(std::vector<double> &force_corr,
                                       std::vector<double> &virial_corr,
                                       const std::vector<double> &v_per_atom);
  void computew_charge_response_cached(std::vector<float> &force_corr,
                                       std::vector<float> &virial_corr,
                                       const std::vector<float> &v_per_atom);
  // ── Tier-2 fused path: energy-only+charge forward (no extended_force) + one combined backward ──
  void set_charge_only_forward(bool b) { charge_only_forward_ = b; }
  template <typename VALUETYPE>
  void compute_combined_response(std::vector<VALUETYPE> &force_total,
                                 std::vector<VALUETYPE> &virial_total,
                                 const std::vector<VALUETYPE> &v_per_atom);
  void computew_combined_response(std::vector<double> &force_total,
                                  std::vector<double> &virial_total,
                                  const std::vector<double> &v_per_atom);
  void computew_combined_response(std::vector<float> &force_total,
                                  std::vector<float> &virial_total,
                                  const std::vector<float> &v_per_atom);
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<double>& force,
                           std::vector<double>& virial,
                           std::vector<double>& atom_energy,
                           std::vector<double>& atom_virial,
                           const int& nframes,
                           const std::vector<double>& coord,
                           const std::vector<int>& atype,
                           const std::vector<double>& box,
                           const std::vector<double>& fparam,
                           const std::vector<double>& aparam,
                           const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
                           std::vector<float>& force,
                           std::vector<float>& virial,
                           std::vector<float>& atom_energy,
                           std::vector<float>& atom_virial,
                           const int& nframes,
                           const std::vector<float>& coord,
                           const std::vector<int>& atype,
                           const std::vector<float>& box,
                           const std::vector<float>& fparam,
                           const std::vector<float>& aparam,
                           const bool atomic);

 private:
  int num_intra_nthreads, num_inter_nthreads;
  bool inited;
  int ntypes;
  int ntypes_spin;
  int dfparam;
  int daparam;
  bool aparam_nall;
  bool has_default_fparam_;
  // copy neighbor list info from host
  torch::jit::script::Module module;
  double rcut;
  NeighborListData nlist_data;
  int max_num_neighbors;
  int gpu_id;
  bool do_message_passing;  // 1:dpa2 model 0:others
  bool gpu_enabled;
  at::Tensor firstneigh_tensor;
  c10::optional<torch::Tensor> mapping_tensor;
  torch::Dict<std::string, torch::Tensor> comm_dict;
  // ── Charge-response graph retention (fix sog/response fusion) ──
  // When retain_charge_graph_ is set (by pair_deepmd for a registered
  // fix sog/response), compute_with_charge keeps the forward autograd graph
  // (coord leaf + latent_charge node) alive so compute_charge_response_cached
  // can do only the VJP backward with seed v_i, instead of a redundant forward.
  bool retain_charge_graph_ = false;
  bool charge_graph_valid_ = false;
  bool charge_only_forward_ = false;     // Tier-2: energy-only+charge forward, defer force to the fix
  torch::Tensor cached_coord_leaf_;      // owning coord leaf, requires_grad
  torch::Tensor cached_latent_charge_;   // [1, nloc, nq] graph node
  torch::Tensor cached_energy_;          // reduced short-range energy (graph node), for the combined seed
  std::vector<int> cached_bkw_map_;
  int cached_fwd_size_ = 0;              // = fwd_map.size() (nall incl. virtual)
  int cached_nall_real_ = 0;
  int cached_nloc_ = 0;
  std::vector<std::vector<int>> remapped_sendlist;
  std::vector<int*> remapped_sendlist_ptrs;
  std::vector<int> remapped_sendnum;
  std::vector<int> remapped_recvnum;
  bool profiler_enabled{false};
  std::string profiler_file;
  /**
   * @brief Translate PyTorch exceptions to the DeePMD-kit exception.
   * @param[in] f The function to run.
   * @example translate_error([&](){...});
   */
  void translate_error(std::function<void()> f);
};

}  // namespace deepmd
