// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "DeepPotPT.h"

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/runtime/jit_exception.h>

#include <cstdint>

#include "common.h"
#include "commonPT.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

void DeepPotPT::translate_error(std::function<void()> f) {
  try {
    f();
    // it seems that libtorch may throw different types of exceptions which are
    // inherbited from different base classes
    // https://github.com/pytorch/pytorch/blob/13316a8d4642454012d34da0d742f1ba93fc0667/torch/csrc/jit/runtime/interpreter.cpp#L924-L939
  } catch (const c10::Error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  } catch (const torch::jit::JITException& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend JIT error: " +
                                   std::string(e.what()));
  } catch (const std::runtime_error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  }
}

torch::Tensor createNlistTensor(const std::vector<std::vector<int>>& data) {
  size_t total_size = 0;
  for (const auto& row : data) {
    total_size += row.size();
  }
  std::vector<int> flat_data;
  flat_data.reserve(total_size);
  for (const auto& row : data) {
    flat_data.insert(flat_data.end(), row.begin(), row.end());
  }

  torch::Tensor flat_tensor = torch::tensor(flat_data, torch::kInt32);
  int nloc = data.size();
  int nnei = nloc > 0 ? total_size / nloc : 0;
  return flat_tensor.view({1, nloc, nnei});
}
DeepPotPT::DeepPotPT() : inited(false) {}
DeepPotPT::DeepPotPT(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false) {
  try {
    translate_error([&] { init(model, gpu_rank, file_content); });
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    throw;
  }
}
void DeepPotPT::init(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  deepmd::load_op_library();
  int gpu_num = torch::cuda::device_count();
  gpu_id = (gpu_num > 0) ? (gpu_rank % gpu_num) : 0;
  gpu_enabled = torch::cuda::is_available();
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
    std::cout << "load model from: " << model << " to cpu " << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  // Configure PyTorch profiler
  const char* env_profiler = std::getenv("DP_PROFILER");
  if (env_profiler && *env_profiler) {
    using torch::profiler::impl::ActivityType;
    using torch::profiler::impl::ExperimentalConfig;
    using torch::profiler::impl::ProfilerConfig;
    using torch::profiler::impl::ProfilerState;
    std::set<ActivityType> activities{ActivityType::CPU};
    if (gpu_enabled) {
      activities.insert(ActivityType::CUDA);
    }
    profiler_file = std::string(env_profiler);
    if (gpu_enabled) {
      profiler_file += "_gpu" + std::to_string(gpu_id);
    }
    profiler_file += ".json";
    ExperimentalConfig exp_cfg;
    ProfilerConfig cfg(ProfilerState::KINETO,
                       false,  // report_input_shapes
                       false,  // profile_memory
                       true,   // with_stack
                       false,  // with_flops
                       true,   // with_modules
                       exp_cfg);
    torch::autograd::profiler::prepareProfiler(cfg, activities);
    torch::autograd::profiler::enableProfiler(cfg, activities);
    std::cout << "PyTorch profiler enabled, output file: " << profiler_file
              << std::endl;
    profiler_enabled = true;
  }
  std::unordered_map<std::string, std::string> metadata = {{"type", ""}};
  module = torch::jit::load(model, device, metadata);
  module.eval();
  do_message_passing = module.run_method("has_message_passing").toBool();
  torch::jit::FusionStrategy strategy;
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
  torch::jit::setFusionStrategy(strategy);

  get_env_nthreads(num_intra_nthreads,
                   num_inter_nthreads);  // need to be fixed as
                                         // DP_INTRA_OP_PARALLELISM_THREADS
  if (num_inter_nthreads) {
    try {
      at::set_num_interop_threads(num_inter_nthreads);
    } catch (...) {
    }
  }
  if (num_intra_nthreads) {
    try {
      at::set_num_threads(num_intra_nthreads);
    } catch (...) {
    }
  }

  auto rcut_ = module.run_method("get_rcut").toDouble();
  rcut = static_cast<double>(rcut_);
  ntypes = module.run_method("get_ntypes").toInt();
  ntypes_spin = 0;
  dfparam = module.run_method("get_dim_fparam").toInt();
  daparam = module.run_method("get_dim_aparam").toInt();
  aparam_nall = module.run_method("is_aparam_nall").toBool();
  if (module.find_method("has_default_fparam")) {
    has_default_fparam_ = module.run_method("has_default_fparam").toBool();
  } else {
    has_default_fparam_ = false;
  }
  inited = true;
}

DeepPotPT::~DeepPotPT() {
  if (profiler_enabled) {
    auto result = torch::autograd::profiler::disableProfiler();
    if (result) {
      result->save(profiler_file);
    }
    std::cout << "PyTorch profiler result saved to " << profiler_file
              << std::endl;
  }
}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute(ENERGYVTYPE& ener,
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
                        const bool atomic) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int32_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  // Detect whether any NULL-type atoms were filtered out.
  bool has_null_atoms = (nall_real < nall);
  int nframes = 1;
  // Reset any retained charge-response graph from the previous step
  // (single-graph lifetime); rebuilt below when retain_charge_graph_ is set.
  if (retain_charge_graph_) {
    charge_graph_valid_ = false;
    cached_coord_leaf_ = torch::Tensor();
    cached_latent_charge_ = torch::Tensor();
    cached_energy_ = torch::Tensor();
  }
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  if (retain_charge_graph_) {
    // Own the coord data as a graph leaf: from_blob aliases the local
    // coord_wrapped vector (dangles after return, esp. on CPU); clone to own it,
    // requires_grad so the model forward builds a graph back to this leaf.
    coord_wrapped_Tensor =
        coord_wrapped_Tensor.detach().clone().requires_grad_(true);
  }
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option).to(device);
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
    if (do_message_passing) {
      if (has_null_atoms) {
        build_comm_dict_with_virtual_atoms(
            comm_dict, lmp_list, fwd_map, remapped_sendlist,
            remapped_sendlist_ptrs, remapped_sendnum, remapped_recvnum);
      } else {
        build_comm_dict(comm_dict, lmp_list, lmp_list.sendlist,
                        lmp_list.sendnum, lmp_list.recvnum);
      }
    }
    if (lmp_list.mapping) {
      std::vector<std::int64_t> mapping(nall_real);
      for (size_t ii = 0; ii < nall_real; ii++) {
        mapping[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
      }
      mapping_tensor =
          torch::from_blob(mapping.data(), {1, nall_real}, int_option)
              .to(device);
    }
  }
  at::Tensor firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);
  bool do_atom_virial_tensor = atomic;
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())}, options)
            .to(device);
  }
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam_.data()),
            {1, lmp_list.inum,
             static_cast<std::int64_t>(aparam_.size()) / lmp_list.inum},
            options)
            .to(device);
  }
  if (do_message_passing) {
    // Keep periodic box in comm_dict up-to-date for lower-level frame
    // corrections (e.g. SOG/LES) that consume comm_dict["box"].
    insert_box_to_comm_dict(comm_dict, box, options, device);
  }
  auto outputs =
      (do_message_passing)
          ? module
                .run_method("forward_lower", coord_wrapped_Tensor, atype_Tensor,
                            firstneigh_tensor, mapping_tensor, fparam_tensor,
                            aparam_tensor, do_atom_virial_tensor, comm_dict)
                .toGenericDict()
          : module
                .run_method("forward_lower", coord_wrapped_Tensor, atype_Tensor,
                            firstneigh_tensor, mapping_tensor, fparam_tensor,
                            aparam_tensor, do_atom_virial_tensor)
                .toGenericDict();
  c10::IValue energy_ = outputs.at("energy");
  c10::IValue force_ = outputs.at("extended_force");
  c10::IValue virial_ = outputs.at("virial");
  torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
  torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
  ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
              cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
  torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // bkw map
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  if (atomic) {
    c10::IValue atom_virial_ = outputs.at("extended_virial");
    c10::IValue atom_energy_ = outputs.at("atom_energy");
    torch::Tensor flat_atom_energy_ =
        atom_energy_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
    datom_energy.resize(nall_real,
                        0.0);  // resize to nall to be consistenet with TF.
    datom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
    torch::Tensor flat_atom_virial_ =
        atom_virial_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    datom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
  }
}
template void DeepPotPT::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepPotPT::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute_with_charge(
    ENERGYVTYPE& ener,
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
    const bool atomic) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  bool has_null_atoms = (nall_real < nall);
  int nframes = 1;
  // Reset any retained charge-response graph from the previous step
  // (single-graph lifetime); rebuilt below when retain_charge_graph_ is set.
  if (retain_charge_graph_) {
    charge_graph_valid_ = false;
    cached_coord_leaf_ = torch::Tensor();
    cached_latent_charge_ = torch::Tensor();
    cached_energy_ = torch::Tensor();
  }
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  if (retain_charge_graph_) {
    // Own the coord data as a graph leaf: from_blob aliases the local
    // coord_wrapped vector (dangles after return, esp. on CPU); clone to own it,
    // requires_grad so the model forward builds a graph back to this leaf.
    coord_wrapped_Tensor =
        coord_wrapped_Tensor.detach().clone().requires_grad_(true);
  }
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option).to(device);
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
    if (do_message_passing) {
      if (has_null_atoms) {
        build_comm_dict_with_virtual_atoms(
            comm_dict, lmp_list, fwd_map, remapped_sendlist,
            remapped_sendlist_ptrs, remapped_sendnum, remapped_recvnum);
      } else {
        build_comm_dict(comm_dict, lmp_list, lmp_list.sendlist,
                        lmp_list.sendnum, lmp_list.recvnum);
      }
    }
    if (lmp_list.mapping) {
      std::vector<std::int64_t> mapping(nall_real);
      for (size_t ii = 0; ii < nall_real; ii++) {
        mapping[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
      }
      mapping_tensor =
          torch::from_blob(mapping.data(), {1, nall_real}, int_option)
              .to(device);
    }
  }
  at::Tensor firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);
  bool do_atom_virial_tensor = atomic;
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())}, options)
            .to(device);
  }
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam_.data()),
            {1, lmp_list.inum,
             static_cast<std::int64_t>(aparam_.size()) / lmp_list.inum},
            options)
            .to(device);
  }
  if (do_message_passing) {
    insert_box_to_comm_dict(comm_dict, box, options, device);
  }
  torch::Tensor cached_energy_node;  // differentiable reduced energy (Tier-2 combined-backward seed)
  auto outputs = [&]() {
    if (charge_only_forward_) {
      // Tier-2 fused path: energy-only + charge forward — NO extended_force.
      // forward_lower_energy_charge(coord, atype, nlist, mapping, comm_dict).
      return (do_message_passing)
                 ? module
                       .run_method("forward_lower_energy_charge",
                                   coord_wrapped_Tensor, atype_Tensor,
                                   firstneigh_tensor, mapping_tensor, comm_dict)
                       .toGenericDict()
                 : module
                       .run_method("forward_lower_energy_charge",
                                   coord_wrapped_Tensor, atype_Tensor,
                                   firstneigh_tensor, mapping_tensor)
                       .toGenericDict();
    }
    return (do_message_passing)
               ? module
                     .run_method("forward_lower", coord_wrapped_Tensor,
                                 atype_Tensor, firstneigh_tensor, mapping_tensor,
                                 fparam_tensor, aparam_tensor,
                                 do_atom_virial_tensor, comm_dict)
                     .toGenericDict()
               : module
                     .run_method("forward_lower", coord_wrapped_Tensor,
                                 atype_Tensor, firstneigh_tensor, mapping_tensor,
                                 fparam_tensor, aparam_tensor,
                                 do_atom_virial_tensor)
                     .toGenericDict();
  }();
  if (charge_only_forward_) {
    c10::IValue energy_ = outputs.at("energy");
    cached_energy_node = energy_.toTensor();
    torch::Tensor cpu_energy_ = cached_energy_node.view({-1}).to(torch::kCPU);
    ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
                cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
    // Zero force/virial here — the pair adds these (harmless), the fix supplies the total.
    dforce.assign(static_cast<size_t>(nall_real) * 3, 0);
    virial.assign(9, 0);
  } else {
    c10::IValue energy_ = outputs.at("energy");
    c10::IValue force_ = outputs.at("extended_force");
    c10::IValue virial_ = outputs.at("virial");
    torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
    torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
    ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
                cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
    torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
    dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                  cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
    torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
    virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                  cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());
  }

  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);

  atom_charge.clear();
  int nchannels = 1;
  if (outputs.contains("latent_charge")) {
    torch::Tensor latent_tensor = outputs.at("latent_charge").toTensor();
    if (latent_tensor.dim() != 3 || latent_tensor.size(0) != nframes) {
      throw deepmd::deepmd_exception(
          "Unexpected latent_charge shape from model output.");
    }
    if (latent_tensor.size(2) < 1) {
      throw deepmd::deepmd_exception(
          "latent_charge must have last dimension >= 1 to map to atom->q.");
    }
    // Preserve all latent_charge channels (dim_out_lr).
    // Flatten as [atom0_ch0, atom0_ch1, ..., atom0_chN, atom1_ch0, ...]
    nchannels = static_cast<int>(latent_tensor.size(2));
    torch::Tensor flat_latent_ =
        latent_tensor.contiguous().view({-1}).to(floatType);
    torch::Tensor cpu_latent_ = flat_latent_.to(torch::kCPU);
    std::vector<VALUETYPE> dcharge_local;
    dcharge_local.assign(cpu_latent_.data_ptr<VALUETYPE>(),
                         cpu_latent_.data_ptr<VALUETYPE>() +
                             cpu_latent_.numel());
    if (dcharge_local.size() != static_cast<size_t>(nframes * nloc * nchannels)) {
      throw deepmd::deepmd_exception(
          "latent_charge size is inconsistent with local atom count.");
    }
    std::vector<VALUETYPE> dcharge_nall_real(nall_real * nchannels, 0);
    std::copy(dcharge_local.begin(), dcharge_local.end(),
              dcharge_nall_real.begin());
    atom_charge.resize(static_cast<size_t>(nframes) * fwd_map.size() * nchannels, 0);
    select_map<VALUETYPE>(atom_charge, dcharge_nall_real, bkw_map, nchannels, nframes,
                          fwd_map.size(), nall_real);
    if (retain_charge_graph_) {
      // Stash the live forward graph for compute_charge_response_cached (VJP-only).
      cached_coord_leaf_ = coord_wrapped_Tensor;
      cached_latent_charge_ = latent_tensor;  // [1, nloc, nq], connected to leaf
      cached_energy_ = cached_energy_node;     // Tier-2: reduced energy node (empty in normal mode)
      cached_bkw_map_ = bkw_map;
      cached_fwd_size_ = static_cast<int>(fwd_map.size());
      cached_nall_real_ = nall_real;
      cached_nloc_ = nloc;
      charge_graph_valid_ = true;
    }
  }

  if (atomic) {
    c10::IValue atom_virial_ = outputs.at("extended_virial");
    c10::IValue atom_energy_ = outputs.at("atom_energy");
    torch::Tensor flat_atom_energy_ =
        atom_energy_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
    datom_energy.resize(nall_real, 0.0);
    datom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
    torch::Tensor flat_atom_virial_ =
        atom_virial_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    datom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
  }
}

template <typename VALUETYPE>
void DeepPotPT::compute_charge_response(
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
    const std::vector<VALUETYPE> &aparam) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  std::vector<VALUETYPE> dcoord, dforce, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  int nframes = 1;
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option).to(device);
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
    if (do_message_passing) {
      build_comm_dict(comm_dict, lmp_list, lmp_list.sendlist,
                      lmp_list.sendnum, lmp_list.recvnum);
    }
    if (lmp_list.mapping) {
      std::vector<std::int64_t> mapping(nall_real);
      for (size_t ii = 0; ii < nall_real; ii++) {
        mapping[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
      }
      mapping_tensor =
          torch::from_blob(mapping.data(), {1, nall_real}, int_option)
              .to(device);
    }
  }
  at::Tensor firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE *>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())}, options)
            .to(device);
  }
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE *>(aparam_.data()),
            {1, lmp_list.inum,
             static_cast<std::int64_t>(aparam_.size()) / lmp_list.inum},
            options)
            .to(device);
  }
  if (do_message_passing) {
    insert_box_to_comm_dict(comm_dict, box, options, device);
  }
  // v_per_atom tensor: [1, nloc, nchannels]
  int nchannels = static_cast<int>(v_per_atom.size()) / nloc;
  at::Tensor v_tensor =
      torch::from_blob(const_cast<VALUETYPE *>(v_per_atom.data()),
                       {1, nloc, nchannels}, options)
          .to(device);
  auto outputs =
      (do_message_passing)
          ? module
                .run_method("charge_response_lower", coord_wrapped_Tensor,
                            atype_Tensor, firstneigh_tensor, v_tensor,
                            mapping_tensor, comm_dict)
                .toGenericDict()
          : module
                .run_method("charge_response_lower", coord_wrapped_Tensor,
                            atype_Tensor, firstneigh_tensor, v_tensor,
                            mapping_tensor)
                .toGenericDict();
  // Extract charge_response_force [1, nall, 3]
  c10::IValue cr_force_ = outputs.at("charge_response_force");
  torch::Tensor flat_crf_ = cr_force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_crf_ = flat_crf_.to(torch::kCPU);
  dforce.assign(cpu_crf_.data_ptr<VALUETYPE>(),
                cpu_crf_.data_ptr<VALUETYPE>() + cpu_crf_.numel());
  force_corr.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force_corr, dforce, bkw_map, 3, nframes,
                        fwd_map.size(), nall_real);
  // Extract charge_response_virial [1, 9]
  c10::IValue cr_virial_ = outputs.at("charge_response_virial");
  torch::Tensor flat_crv_ = cr_virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_crv_ = flat_crv_.to(torch::kCPU);
  dforce.assign(cpu_crv_.data_ptr<VALUETYPE>(),
                cpu_crv_.data_ptr<VALUETYPE>() + cpu_crv_.numel());
  virial_corr.resize(9, 0.0);
  for (int f = 0; f < nframes; ++f)
    for (int i = 0; i < 9; ++i)
      virial_corr[i] += dforce[f * 9 + i];
}

// ── Cached charge-response: reuse the pair's retained forward graph, do only
//    the VJP backward seeded by v_i (no second forward_lower/charge_response_lower).
//    Mathematically identical to compute_charge_response (sog_model.py:402-450).
template <typename VALUETYPE>
void DeepPotPT::compute_charge_response_cached(
    std::vector<VALUETYPE> &force_corr,
    std::vector<VALUETYPE> &virial_corr,
    const std::vector<VALUETYPE> &v_per_atom) {
  if (!charge_graph_valid_ || !cached_coord_leaf_.defined() ||
      !cached_latent_charge_.defined()) {
    throw deepmd::deepmd_exception(
        "compute_charge_response_cached: no valid retained forward graph "
        "(pair latent-charge forward did not run this step).");
  }
  const int nloc = cached_nloc_;
  const int nchannels =
      (nloc > 0) ? static_cast<int>(v_per_atom.size()) / nloc : 1;
  torch::ScalarType floatType =
      std::is_same<VALUETYPE, float>::value ? torch::kFloat32 : torch::kFloat64;
  // seed v_i (per-atom potential) onto the retained latent_charge (real order)
  at::Tensor v_tensor =
      torch::from_blob(const_cast<VALUETYPE *>(v_per_atom.data()),
                       {1, nloc, nchannels},
                       torch::TensorOptions().dtype(floatType))
          .to(cached_coord_leaf_.device())
          .to(cached_coord_leaf_.dtype());
  at::Tensor q_loc = cached_latent_charge_.narrow(1, 0, nloc);
  at::Tensor scalar = (v_tensor * q_loc).sum();
  auto grads = torch::autograd::grad({scalar}, {cached_coord_leaf_},
                                     /*grad_outputs=*/{torch::ones_like(scalar)},
                                     /*retain_graph=*/true,
                                     /*create_graph=*/false);
  at::Tensor force_ext = -grads[0];  // [1, nall_real, 3]
  at::Tensor virial =
      torch::einsum("fak,faj->fkj", {force_ext, cached_coord_leaf_})
          .reshape({9});
  // ── map extended force back to LAMMPS order (mirror compute_charge_response) ──
  torch::Tensor cpu_f =
      force_ext.reshape({-1}).to(floatType).to(torch::kCPU).contiguous();
  std::vector<VALUETYPE> dforce(
      cpu_f.data_ptr<VALUETYPE>(),
      cpu_f.data_ptr<VALUETYPE>() + cpu_f.numel());
  force_corr.resize(static_cast<size_t>(cached_fwd_size_) * 3);
  select_map<VALUETYPE>(force_corr, dforce, cached_bkw_map_, 3, 1,
                        cached_fwd_size_, cached_nall_real_);
  torch::Tensor cpu_v = virial.to(floatType).to(torch::kCPU).contiguous();
  virial_corr.assign(cpu_v.data_ptr<VALUETYPE>(),
                     cpu_v.data_ptr<VALUETYPE>() + cpu_v.numel());
}

void DeepPotPT::computew_charge_response_cached(
    std::vector<double> &force_corr, std::vector<double> &virial_corr,
    const std::vector<double> &v_per_atom) {
  translate_error([&] {
    compute_charge_response_cached(force_corr, virial_corr, v_per_atom);
  });
}
void DeepPotPT::computew_charge_response_cached(
    std::vector<float> &force_corr, std::vector<float> &virial_corr,
    const std::vector<float> &v_per_atom) {
  translate_error([&] {
    compute_charge_response_cached(force_corr, virial_corr, v_per_atom);
  });
}

// ── Tier-2 combined backward: one grad( E_short + Σ_i v_i q_i , coord_leaf ) on the retained
//    energy-only+charge graph -> TOTAL (short-range + charge-response) force + virial. Replaces the
//    pair's internal force backward AND the fix's separate VJP with a single backward pass. ──
template <typename VALUETYPE>
void DeepPotPT::compute_combined_response(
    std::vector<VALUETYPE> &force_total,
    std::vector<VALUETYPE> &virial_total,
    const std::vector<VALUETYPE> &v_per_atom) {
  if (!charge_graph_valid_ || !cached_coord_leaf_.defined() ||
      !cached_latent_charge_.defined() || !cached_energy_.defined()) {
    throw deepmd::deepmd_exception(
        "compute_combined_response: no valid retained energy+charge graph "
        "(pair energy-only forward did not run this step / not in fused mode).");
  }
  const int nloc = cached_nloc_;
  const int nchannels =
      (nloc > 0) ? static_cast<int>(v_per_atom.size()) / nloc : 1;
  torch::ScalarType floatType =
      std::is_same<VALUETYPE, float>::value ? torch::kFloat32 : torch::kFloat64;
  at::Tensor v_tensor =
      torch::from_blob(const_cast<VALUETYPE *>(v_per_atom.data()),
                       {1, nloc, nchannels},
                       torch::TensorOptions().dtype(floatType))
          .to(cached_coord_leaf_.device())
          .to(cached_coord_leaf_.dtype());
  at::Tensor q_loc = cached_latent_charge_.narrow(1, 0, nloc);
  // total force = -∂( E_short + Σ_i v_i q_i )/∂r  (short-range + charge-response, one backward)
  at::Tensor scalar = cached_energy_.sum() + (v_tensor * q_loc).sum();
  auto grads = torch::autograd::grad({scalar}, {cached_coord_leaf_},
                                     /*grad_outputs=*/{torch::ones_like(scalar)},
                                     /*retain_graph=*/false,
                                     /*create_graph=*/false);
  at::Tensor force_ext = -grads[0];  // [1, nall_real, 3]
  at::Tensor virial =
      torch::einsum("fak,faj->fkj", {force_ext, cached_coord_leaf_})
          .reshape({9});
  torch::Tensor cpu_f =
      force_ext.reshape({-1}).to(floatType).to(torch::kCPU).contiguous();
  std::vector<VALUETYPE> dforce(
      cpu_f.data_ptr<VALUETYPE>(),
      cpu_f.data_ptr<VALUETYPE>() + cpu_f.numel());
  force_total.resize(static_cast<size_t>(cached_fwd_size_) * 3);
  select_map<VALUETYPE>(force_total, dforce, cached_bkw_map_, 3, 1,
                        cached_fwd_size_, cached_nall_real_);
  torch::Tensor cpu_v = virial.to(floatType).to(torch::kCPU).contiguous();
  virial_total.assign(cpu_v.data_ptr<VALUETYPE>(),
                      cpu_v.data_ptr<VALUETYPE>() + cpu_v.numel());
  charge_graph_valid_ = false;  // graph consumed (freed by retain_graph=false)
}

void DeepPotPT::computew_combined_response(
    std::vector<double> &force_total, std::vector<double> &virial_total,
    const std::vector<double> &v_per_atom) {
  translate_error([&] {
    compute_combined_response(force_total, virial_total, v_per_atom);
  });
}
void DeepPotPT::computew_combined_response(
    std::vector<float> &force_total, std::vector<float> &virial_total,
    const std::vector<float> &v_per_atom) {
  translate_error([&] {
    compute_combined_response(force_total, virial_total, v_per_atom);
  });
}

template void DeepPotPT::compute_charge_response_cached<double>(
    std::vector<double> &force_corr, std::vector<double> &virial_corr,
    const std::vector<double> &v_per_atom);
template void DeepPotPT::compute_charge_response_cached<float>(
    std::vector<float> &force_corr, std::vector<float> &virial_corr,
    const std::vector<float> &v_per_atom);

template void DeepPotPT::compute_combined_response<double>(
    std::vector<double> &force_total, std::vector<double> &virial_total,
    const std::vector<double> &v_per_atom);
template void DeepPotPT::compute_combined_response<float>(
    std::vector<float> &force_total, std::vector<float> &virial_total,
    const std::vector<float> &v_per_atom);

template void DeepPotPT::compute_charge_response<double>(
    std::vector<double> &force_corr, std::vector<double> &virial_corr,
    const std::vector<double> &v_per_atom,
    const std::vector<double> &coord, const std::vector<int> &atype,
    const std::vector<double> &box, const int nghost,
    const InputNlist &lmp_list, const int &ago,
    const std::vector<double> &fparam, const std::vector<double> &aparam);
template void DeepPotPT::compute_charge_response<float>(
    std::vector<float> &force_corr, std::vector<float> &virial_corr,
    const std::vector<float> &v_per_atom,
    const std::vector<float> &coord, const std::vector<int> &atype,
    const std::vector<float> &box, const int nghost,
    const InputNlist &lmp_list, const int &ago,
    const std::vector<float> &fparam, const std::vector<float> &aparam);

template void DeepPotPT::compute_with_charge<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    std::vector<double>& atom_charge,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepPotPT::compute_with_charge<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    std::vector<float>& atom_charge,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute(ENERGYVTYPE& ener,
                        std::vector<VALUETYPE>& force,
                        std::vector<VALUETYPE>& virial,
                        std::vector<VALUETYPE>& atom_energy,
                        std::vector<VALUETYPE>& atom_virial,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const std::vector<VALUETYPE>& fparam,
                        const std::vector<VALUETYPE>& aparam,
                        const bool atomic) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  std::vector<VALUETYPE> coord_wrapped = coord;
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  int nframes = 1;
  std::vector<torch::jit::IValue> inputs;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, natoms, 3}, options)
          .to(device);
  inputs.push_back(coord_wrapped_Tensor);
  std::vector<std::int64_t> atype_64(atype.begin(), atype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, natoms}, int_options).to(device);
  inputs.push_back(atype_Tensor);
  c10::optional<torch::Tensor> box_Tensor;
  if (!box.empty()) {
    box_Tensor =
        torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options)
            .to(device);
  }
  inputs.push_back(box_Tensor);
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())}, options)
            .to(device);
  }
  inputs.push_back(fparam_tensor);
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam.data()),
            {1, natoms, static_cast<std::int64_t>(aparam.size()) / natoms},
            options)
            .to(device);
  }
  inputs.push_back(aparam_tensor);
  bool do_atom_virial_tensor = atomic;
  inputs.push_back(do_atom_virial_tensor);
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module.forward(inputs).toGenericDict();
  c10::IValue energy_ = outputs.at("energy");
  c10::IValue force_ = outputs.at("force");
  c10::IValue virial_ = outputs.at("virial");
  torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
  torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
  ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
              cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  force.assign(cpu_force_.data_ptr<VALUETYPE>(),
               cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
  torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());
  if (atomic) {
    c10::IValue atom_virial_ = outputs.at("atom_virial");
    c10::IValue atom_energy_ = outputs.at("atom_energy");
    torch::Tensor flat_atom_energy_ =
        atom_energy_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
    atom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
    torch::Tensor flat_atom_virial_ =
        atom_virial_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    atom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
  }
}

template void DeepPotPT::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
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
template void DeepPotPT::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
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
void DeepPotPT::get_type_map(std::string& type_map) {
  auto ret = module.run_method("get_type_map").toList();
  for (const torch::IValue& element : ret) {
    type_map += torch::str(element);  // Convert each element to a string
    type_map += " ";                  // Add a space between elements
  }
}

// forward to template method
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam,
                         const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            fparam, aparam, atomic);
  });
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam,
                         const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            fparam, aparam, atomic);
  });
}
void DeepPotPT::computew(std::vector<double>& ener,
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
                         const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            nghost, inlist, ago, fparam, aparam, atomic);
  });
}
void DeepPotPT::computew_with_charge(std::vector<double>& ener,
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
                                     const bool atomic) {
  translate_error([&] {
    compute_with_charge(ener, force, virial, atom_energy, atom_virial,
                        atom_charge, coord, atype, box, nghost, inlist, ago,
                        fparam, aparam, atomic);
  });
}
void DeepPotPT::computew_with_charge(std::vector<double>& ener,
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
                                     const bool atomic) {
  translate_error([&] {
    compute_with_charge(ener, force, virial, atom_energy, atom_virial,
                        atom_charge, coord, atype, box, nghost, inlist, ago,
                        fparam, aparam, atomic);
  });
}
void DeepPotPT::computew_charge_response(
    std::vector<double> &force_corr, std::vector<double> &virial_corr,
    const std::vector<double> &v_per_atom,
    const std::vector<double> &coord, const std::vector<int> &atype,
    const std::vector<double> &box, const int nghost,
    const InputNlist &inlist, const int &ago,
    const std::vector<double> &fparam,
    const std::vector<double> &aparam) {
  translate_error([&] {
    compute_charge_response(force_corr, virial_corr, v_per_atom, coord, atype,
                            box, nghost, inlist, ago, fparam, aparam);
  });
}
void DeepPotPT::computew_charge_response(
    std::vector<float> &force_corr, std::vector<float> &virial_corr,
    const std::vector<float> &v_per_atom,
    const std::vector<float> &coord, const std::vector<int> &atype,
    const std::vector<float> &box, const int nghost,
    const InputNlist &inlist, const int &ago,
    const std::vector<float> &fparam,
    const std::vector<float> &aparam) {
  translate_error([&] {
    compute_charge_response(force_corr, virial_corr, v_per_atom, coord, atype,
                            box, nghost, inlist, ago, fparam, aparam);
  });
}
void DeepPotPT::computew(std::vector<double>& ener,
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
                         const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            nghost, inlist, ago, fparam, aparam, atomic);
  });
}
void DeepPotPT::computew_mixed_type(std::vector<double>& ener,
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
                                    const bool atomic) {
  throw deepmd::deepmd_exception("computew_mixed_type is not implemented");
}
void DeepPotPT::computew_mixed_type(std::vector<double>& ener,
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
                                    const bool atomic) {
  throw deepmd::deepmd_exception("computew_mixed_type is not implemented");
}
#endif
