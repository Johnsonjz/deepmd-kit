#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>
#include <cuda_runtime.h>
#include "cufinufft_hdrs/cufinufft_dl.h"

void launch_sog_modulate_kernel(
    int nkx, int nky, int nkz,
    int num_bandwidth,
    const double* coeff_ptr,
    const double* amp_ptr,
    const double* bw_sq_ptr,
    const double* cell_inv_ptr,
    double* out_corr_ptr,
    double* out_kfac_sum_ptr,
    cudaStream_t stream
);

void launch_sog_modulate_kernel(
    int nkx, int nky, int nkz,
    int num_bandwidth,
    const float* coeff_ptr,
    const float* amp_ptr,
    const float* bw_sq_ptr,
    const float* cell_inv_ptr,
    float* out_corr_ptr,
    float* out_kfac_sum_ptr,
    cudaStream_t stream
);

void launch_sog_build_gradconv_kernel(
    int nkx,
    int nky,
    int nkz,
    int num_bandwidth,
    const double* coeff_ptr,
    const double* amp_ptr,
    const double* bw_sq_ptr,
    const double* cell_inv_ptr,
    double* gradconv_x_ptr,
    double* gradconv_y_ptr,
    double* gradconv_z_ptr,
    cudaStream_t stream
);

void launch_sog_build_gradconv_kernel(
    int nkx,
    int nky,
    int nkz,
    int num_bandwidth,
    const float* coeff_ptr,
    const float* amp_ptr,
    const float* bw_sq_ptr,
    const float* cell_inv_ptr,
    float* gradconv_x_ptr,
    float* gradconv_y_ptr,
    float* gradconv_z_ptr,
    cudaStream_t stream
);

void launch_sog_backward_energy_kernel(
    int nkx,
    int nky,
    int nkz,
    int num_bandwidth,
    const double* coeff_ptr,
    const double* amp_ptr,
    const double* bw_sq_ptr,
    const double* cell_inv_ptr,
    double* conv_grid_ptr,
    double* sum_base_rho_ptr,
    double* sum_base_ptr,
    double* sum_dbw_rho_ptr,
    double* sum_dbw_ptr,
    cudaStream_t stream
);

void launch_sog_backward_energy_kernel(
    int nkx,
    int nky,
    int nkz,
    int num_bandwidth,
    const float* coeff_ptr,
    const float* amp_ptr,
    const float* bw_sq_ptr,
    const float* cell_inv_ptr,
    float* conv_grid_ptr,
    float* sum_base_rho_ptr,
    float* sum_base_ptr,
    float* sum_dbw_rho_ptr,
    float* sum_dbw_ptr,
    cudaStream_t stream
);

void launch_sog_backward_force_prepare_kernel(
    int nkx,
    int nky,
    int nkz,
    int num_bandwidth,
    const double* rho_grid_ptr,
    const double* sminus_x_grid_ptr,
    const double* sminus_y_grid_ptr,
    const double* sminus_z_grid_ptr,
    const double* amp_ptr,
    const double* bw_sq_ptr,
    const double* cell_inv_ptr,
    double inv_volume,
    double* sum_base_force_ptr,
    double* sum_dbw_force_ptr,
    double* indirect_x_ptr,
    double* indirect_y_ptr,
    double* indirect_z_ptr,
    cudaStream_t stream
);

void launch_sog_backward_force_prepare_kernel(
    int nkx,
    int nky,
    int nkz,
    int num_bandwidth,
    const float* rho_grid_ptr,
    const float* sminus_x_grid_ptr,
    const float* sminus_y_grid_ptr,
    const float* sminus_z_grid_ptr,
    const float* amp_ptr,
    const float* bw_sq_ptr,
    const float* cell_inv_ptr,
    float inv_volume,
    float* sum_base_force_ptr,
    float* sum_dbw_force_ptr,
    float* indirect_x_ptr,
    float* indirect_y_ptr,
    float* indirect_z_ptr,
    cudaStream_t stream
);

struct PlanKey {
    int nkx, nky, nkz;
    int type;
    int precision;

    bool operator==(const PlanKey& other) const {
        return nkx == other.nkx && nky == other.nky && nkz == other.nkz && type == other.type &&
               precision == other.precision;
    }
};

namespace std {
    template <>
    struct hash<PlanKey> {
        size_t operator()(const PlanKey& k) const {
            return (hash<int>()(k.nkx) ^ (hash<int>()(k.nky) << 1)) ^
                   (hash<int>()(k.nkz) << 2) ^ (hash<int>()(k.type) << 3) ^
                   (hash<int>()(k.precision) << 4);
        }
    };
}

static std::unordered_map<PlanKey, cufinufft_plan> g_plan_cache;
static std::mutex g_plan_mutex;

template <typename scalar_t>
cufinufft_plan get_or_create_plan(int nkx, int nky, int nkz, int type) {
    constexpr int precision = std::is_same_v<scalar_t, float> ? 0 : 1;
    PlanKey key = {nkx, nky, nkz, type, precision};
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    if (g_plan_cache.find(key) != g_plan_cache.end()) {
        return g_plan_cache[key];
    }
    
    auto& finufft = CuFinufftLib::getInstance();
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (finufft.makeplanf == nullptr) {
            throw std::runtime_error("cufinufftf_makeplan symbol is not available.");
        }
    } else {
        if (finufft.makepland == nullptr) {
            throw std::runtime_error("cufinufft_makeplan symbol is not available.");
        }
    }

    cufinufft_opts opts;
    cufinufft_opts* opts_ptr = nullptr;
    bool have_default_opts = false;
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (finufft.default_optsf != nullptr) {
            finufft.default_optsf(&opts);
            have_default_opts = true;
        } else if (finufft.default_optsd != nullptr) {
            // Fallback to double default opts when float symbol is unavailable.
            finufft.default_optsd(&opts);
            have_default_opts = true;
        }
    } else {
        if (finufft.default_optsd != nullptr) {
            finufft.default_optsd(&opts);
            have_default_opts = true;
        } else if (finufft.default_optsf != nullptr) {
            finufft.default_optsf(&opts);
            have_default_opts = true;
        }
    }
    if (have_default_opts) {
        opts.modeord = 1;
        opts_ptr = &opts;
    }
    
    cufinufft_plan plan;
    int64_t nmodes[3] = { (2 * nkx + 1), (2 * nky + 1), (2 * nkz + 1) };
    
    // Type-1 uses isign=-1; Type-2 uses isign=+1 to match pytorch_finufft path.
    int isign = (type == 2) ? 1 : -1;
    int ier = 0;
    if constexpr (std::is_same_v<scalar_t, float>) {
        ier = finufft.makeplanf(type, 3, nmodes, isign, 1, static_cast<float>(1e-4), &plan, opts_ptr);
    } else {
        ier = finufft.makepland(type, 3, nmodes, isign, 1, 1e-4, &plan, opts_ptr);
    }
    if (ier != 0) {
        throw std::runtime_error("cufinufft_makeplan failed.");
    }
    
    g_plan_cache[key] = plan;
    return plan;
}

template <typename scalar_t>
constexpr at::ScalarType complex_dtype_of() {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return at::kComplexFloat;
    } else {
        return at::kComplexDouble;
    }
}

template <typename scalar_t>
scalar_t point_limit_of() {
    return static_cast<scalar_t>(M_PI) - static_cast<scalar_t>(32.0) * std::numeric_limits<scalar_t>::epsilon();
}

template <typename scalar_t>
scalar_t two_pi_of() {
    return static_cast<scalar_t>(2.0 * M_PI);
}

template <typename scalar_t>
int finufft_setpts(cufinufft_plan plan, int64_t npts, scalar_t* x, scalar_t* y, scalar_t* z) {
    auto& finufft = CuFinufftLib::getInstance();
    if constexpr (std::is_same_v<scalar_t, float>) {
        TORCH_CHECK(finufft.setptsf != nullptr, "cufinufftf_setpts symbol is not available");
        return finufft.setptsf(plan, npts, x, y, z, 0, nullptr, nullptr, nullptr);
    } else {
        TORCH_CHECK(finufft.setptsd != nullptr, "cufinufft_setpts symbol is not available");
        return finufft.setptsd(plan, npts, x, y, z, 0, nullptr, nullptr, nullptr);
    }
}

template <typename scalar_t>
int finufft_execute(cufinufft_plan plan, scalar_t* c, scalar_t* fk) {
    auto& finufft = CuFinufftLib::getInstance();
    if constexpr (std::is_same_v<scalar_t, float>) {
        TORCH_CHECK(finufft.executef != nullptr, "cufinufftf_execute symbol is not available");
        return finufft.executef(plan, c, fk);
    } else {
        TORCH_CHECK(finufft.executed != nullptr, "cufinufft_execute symbol is not available");
        return finufft.executed(plan, c, fk);
    }
}

std::vector<at::Tensor> nufft_sog_frame_correction_bundle(
    const at::Tensor& coord,
    const at::Tensor& latent_charge, // complex
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    bool need_force,
    bool need_virial
);

std::vector<at::Tensor> nufft_sog_frame_correction_backward_energy(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr
);

std::vector<at::Tensor> nufft_sog_frame_correction_backward_bundle(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr,
    const at::Tensor& grad_force,
    const at::Tensor& force_forward,
    bool need_latent_grad
);

namespace {

void check_sog_dtype_supported(const at::Tensor& t, const char* name) {
    TORCH_CHECK(
        t.scalar_type() == at::kDouble || t.scalar_type() == at::kFloat,
        name,
        " must be float32 or float64"
    );
}

void check_sog_same_dtype(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const at::Tensor* grad_corr = nullptr,
    const at::Tensor* grad_force = nullptr,
    const at::Tensor* force_forward = nullptr
) {
    auto dtype = coord.scalar_type();
    TORCH_CHECK(latent_charge.scalar_type() == dtype, "latent_charge dtype must match coord dtype");
    TORCH_CHECK(box.scalar_type() == dtype, "box dtype must match coord dtype");
    TORCH_CHECK(amp.scalar_type() == dtype, "amp dtype must match coord dtype");
    TORCH_CHECK(bandwidth.scalar_type() == dtype, "bandwidth dtype must match coord dtype");
    if (grad_corr != nullptr) {
        TORCH_CHECK(grad_corr->scalar_type() == dtype, "grad_corr dtype must match coord dtype");
    }
    if (grad_force != nullptr) {
        TORCH_CHECK(grad_force->scalar_type() == dtype, "grad_force dtype must match coord dtype");
    }
    if (force_forward != nullptr) {
        TORCH_CHECK(force_forward->scalar_type() == dtype, "force_forward dtype must match coord dtype");
    }
}

}  // namespace

template <typename scalar_t>
static std::vector<at::Tensor> nufft_sog_frame_correction_bundle_impl(
    const at::Tensor& coord,
    const at::Tensor& latent_charge, // complex
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    bool need_force,
    bool need_virial
) {
    static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>);
    using complex_t = c10::complex<scalar_t>;

    TORCH_CHECK(coord.is_cuda(), "coord must be a CUDA tensor");
    TORCH_CHECK(latent_charge.is_cuda(), "latent_charge must be a CUDA tensor");
    TORCH_CHECK(box.is_cuda(), "box must be a CUDA tensor");
    TORCH_CHECK(amp.is_cuda(), "amp must be a CUDA tensor");
    TORCH_CHECK(bandwidth.is_cuda(), "bandwidth must be a CUDA tensor");
    constexpr at::ScalarType expected_dtype = std::is_same_v<scalar_t, float> ? at::kFloat : at::kDouble;
    TORCH_CHECK(coord.scalar_type() == expected_dtype, "coord dtype mismatch with kernel type");
    TORCH_CHECK(latent_charge.scalar_type() == expected_dtype, "latent_charge dtype mismatch with kernel type");
    TORCH_CHECK(box.scalar_type() == expected_dtype, "box dtype mismatch with kernel type");
    TORCH_CHECK(amp.scalar_type() == expected_dtype, "amp dtype mismatch with kernel type");
    TORCH_CHECK(bandwidth.scalar_type() == expected_dtype, "bandwidth dtype mismatch with kernel type");
    TORCH_CHECK(coord.dim() == 3 && coord.size(2) == 3, "coord must be [nf, nloc, 3]");
    TORCH_CHECK(box.dim() == 3 && box.size(1) == 3 && box.size(2) == 3, "box must be [nf, 3, 3]");
    
    int nf = coord.size(0);
    int nloc = coord.size(1);
    TORCH_CHECK(nk_cpu.size() == static_cast<size_t>(nf * 3), "nk_cpu length must be nf*3");
    
    auto bw_sq = at::square(bandwidth);
    TORCH_CHECK(amp.numel() == 1, "amp must contain exactly one scalar value");
    int num_bandwidth = static_cast<int>(bw_sq.numel());
    
    auto corr_out = at::zeros({nf, 1}, coord.options());
    auto kfac_sum_out = at::zeros({nf, 1}, coord.options());
    auto force_out = need_force ? at::zeros({nf, nloc, 3}, coord.options()) : at::Tensor();
    auto virial_out = need_virial ? at::zeros({nf, nloc, 1, 9}, coord.options()) : at::Tensor();

    auto& finufft = CuFinufftLib::getInstance();
    TORCH_CHECK(finufft.handle != nullptr, "Could not load libcufinufft.so");

    auto volumes = at::linalg_det(box);
    auto cells_inv = at::linalg_inv(box);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    for (int f = 0; f < nf; ++f) {
        int nkx = nk_cpu[f * 3 + 0];
        int nky = nk_cpu[f * 3 + 1];
        int nkz = nk_cpu[f * 3 + 2];
        TORCH_CHECK(nkx > 0 && nky > 0 && nkz > 0, "nk_cpu entries must be positive");
        
        cufinufft_plan plan = get_or_create_plan<scalar_t>(nkx, nky, nkz, 1);
        
        // 1. Prepare inputs: r_in (scaled to [-pi, pi])
        auto r_fracs_f = at::einsum("ni,ij->nj", {coord[f], cells_inv[f]});
        auto r_fracs_shifted = at::remainder(
            r_fracs_f + static_cast<scalar_t>(0.5),
            static_cast<scalar_t>(1.0)
        ) - static_cast<scalar_t>(0.5);
        const scalar_t point_limit = point_limit_of<scalar_t>();
        auto r_in = at::clamp(two_pi_of<scalar_t>() * r_fracs_shifted, -point_limit, point_limit).contiguous();
        
        auto q_real = latent_charge[f].reshape({-1}).contiguous();
        TORCH_CHECK(q_real.numel() == nloc, "latent_charge per frame must contain nloc values");
        auto charge_f = at::complex(q_real, at::zeros_like(q_real)).contiguous();

        auto r_in_t = r_in.transpose(0, 1).contiguous();

        // 2. Set points in cuFINUFFT
        int ier_set = finufft_setpts<scalar_t>(
            plan,
            static_cast<int64_t>(nloc),
            r_in_t[0].template data_ptr<scalar_t>(),
            r_in_t[1].template data_ptr<scalar_t>(),
            r_in_t[2].template data_ptr<scalar_t>()
        );
        TORCH_CHECK(ier_set == 0, "cufinufftf_setpts failed.");

        // Allocate buffer for returned complex grid
        int dim_x = 2 * nkx + 1;
        int dim_y = 2 * nky + 1;
        int dim_z = 2 * nkz + 1;
        int total_grid = dim_x * dim_y * dim_z;
        auto grid_out = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));

        // 3. Execute Type-1 NUFFT
        int ier_exec = finufft_execute<scalar_t>(
            plan,
            reinterpret_cast<scalar_t*>(charge_f.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(grid_out.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec == 0, "cufinufftf_execute failed.");

        // 4. Modulate and reduce to compute Energy (corr_redu)
        launch_sog_modulate_kernel(
            nkx, nky, nkz, 
            num_bandwidth,
            reinterpret_cast<const scalar_t*>(grid_out.template data_ptr<complex_t>()),
            amp.template data_ptr<scalar_t>(),
            bw_sq.template data_ptr<scalar_t>(),
            cells_inv[f].template data_ptr<scalar_t>(),
            corr_out[f].template data_ptr<scalar_t>(),
            kfac_sum_out[f].template data_ptr<scalar_t>(),
            stream
        );

        auto volume_f = volumes[f];
        corr_out[f] = corr_out[f] / (2.0 * volume_f);

        if (remove_self_interaction) {
            auto q_sq_sum = at::sum(q_real * q_real);
            corr_out[f] = corr_out[f] - q_sq_sum * (kfac_sum_out[f] / (2.0 * volume_f));
        }

        if (need_force || need_virial) {
            auto grad_conv_x = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_conv_y = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_conv_z = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));

            launch_sog_build_gradconv_kernel(
                nkx,
                nky,
                nkz,
                num_bandwidth,
                reinterpret_cast<const scalar_t*>(grid_out.template data_ptr<complex_t>()),
                amp.template data_ptr<scalar_t>(),
                bw_sq.template data_ptr<scalar_t>(),
                cells_inv[f].template data_ptr<scalar_t>(),
                reinterpret_cast<scalar_t*>(grad_conv_x.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_y.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_z.template data_ptr<complex_t>()),
                stream
            );

            cufinufft_plan plan_t2 = get_or_create_plan<scalar_t>(nkx, nky, nkz, 2);
            int ier_set_t2 = finufft_setpts<scalar_t>(
                plan_t2,
                static_cast<int64_t>(nloc),
                r_in_t[0].template data_ptr<scalar_t>(),
                r_in_t[1].template data_ptr<scalar_t>(),
                r_in_t[2].template data_ptr<scalar_t>()
            );
            TORCH_CHECK(ier_set_t2 == 0, "cufinufftf_setpts(type2) failed.");

            auto grad_field_x = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_field_y = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_field_z = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));

            int ier_exec_t2_x = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(grad_field_x.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_x.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_x == 0, "cufinufftf_execute(type2,x) failed.");

            int ier_exec_t2_y = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(grad_field_y.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_y.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_y == 0, "cufinufftf_execute(type2,y) failed.");

            int ier_exec_t2_z = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(grad_field_z.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_z.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_z == 0, "cufinufftf_execute(type2,z) failed.");

            auto force_prefac = 1.0 / volume_f;
            auto force_x = -(q_real * at::real(grad_field_x)) * force_prefac;
            auto force_y = -(q_real * at::real(grad_field_y)) * force_prefac;
            auto force_z = -(q_real * at::real(grad_field_z)) * force_prefac;
            auto force_frame = at::stack({force_x, force_y, force_z}, 1);

            if (need_force) {
                force_out[f].copy_(force_frame);
            }
            if (need_virial) {
                auto virial_frame = at::einsum("ai,aj->aij", {force_frame, coord[f]})
                                       .reshape({nloc, 1, 9});
                virial_out[f].copy_(virial_frame);
            }
        }
    }

    std::vector<at::Tensor> res;
    res.push_back(corr_out);
    if (need_force) res.push_back(force_out);
    if (need_virial) res.push_back(virial_out);
    return res;
}

template <typename scalar_t>
static std::vector<at::Tensor> nufft_sog_frame_correction_backward_energy_impl(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr
) {
    static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>);
    using complex_t = c10::complex<scalar_t>;

    TORCH_CHECK(coord.is_cuda(), "coord must be a CUDA tensor");
    TORCH_CHECK(latent_charge.is_cuda(), "latent_charge must be a CUDA tensor");
    TORCH_CHECK(box.is_cuda(), "box must be a CUDA tensor");
    TORCH_CHECK(amp.is_cuda(), "amp must be a CUDA tensor");
    TORCH_CHECK(bandwidth.is_cuda(), "bandwidth must be a CUDA tensor");
    TORCH_CHECK(grad_corr.is_cuda(), "grad_corr must be a CUDA tensor");
    constexpr at::ScalarType expected_dtype = std::is_same_v<scalar_t, float> ? at::kFloat : at::kDouble;
    TORCH_CHECK(coord.scalar_type() == expected_dtype, "coord dtype mismatch with kernel type");
    TORCH_CHECK(latent_charge.scalar_type() == expected_dtype, "latent_charge dtype mismatch with kernel type");
    TORCH_CHECK(box.scalar_type() == expected_dtype, "box dtype mismatch with kernel type");
    TORCH_CHECK(amp.scalar_type() == expected_dtype, "amp dtype mismatch with kernel type");
    TORCH_CHECK(bandwidth.scalar_type() == expected_dtype, "bandwidth dtype mismatch with kernel type");
    TORCH_CHECK(grad_corr.scalar_type() == expected_dtype, "grad_corr dtype mismatch with kernel type");

    int nf = coord.size(0);
    int nloc = coord.size(1);
    TORCH_CHECK(nk_cpu.size() == static_cast<size_t>(nf * 3), "nk_cpu length must be nf*3");
    TORCH_CHECK(amp.numel() == 1, "amp must contain exactly one scalar value");
    TORCH_CHECK(grad_corr.dim() == 2 && grad_corr.size(0) == nf && grad_corr.size(1) == 1,
                "grad_corr must be [nf, 1]");

    auto bw_sq = at::square(bandwidth);
    int num_bandwidth = static_cast<int>(bw_sq.numel());

    auto grad_latent = at::zeros_like(latent_charge);
    auto grad_amp = at::zeros_like(amp);
    auto grad_bandwidth = at::zeros_like(bandwidth);

    auto& finufft = CuFinufftLib::getInstance();
    TORCH_CHECK(finufft.handle != nullptr, "Could not load libcufinufft.so");

    auto volumes = at::linalg_det(box);
    auto cells_inv = at::linalg_inv(box);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    for (int f = 0; f < nf; ++f) {
        int nkx = nk_cpu[f * 3 + 0];
        int nky = nk_cpu[f * 3 + 1];
        int nkz = nk_cpu[f * 3 + 2];
        TORCH_CHECK(nkx > 0 && nky > 0 && nkz > 0, "nk_cpu entries must be positive");

        cufinufft_plan plan_t1 = get_or_create_plan<scalar_t>(nkx, nky, nkz, 1);

        auto r_fracs_f = at::einsum("ni,ij->nj", {coord[f], cells_inv[f]});
        auto r_fracs_shifted = at::remainder(
            r_fracs_f + static_cast<scalar_t>(0.5),
            static_cast<scalar_t>(1.0)
        ) - static_cast<scalar_t>(0.5);
        const scalar_t point_limit = point_limit_of<scalar_t>();
        auto r_in = at::clamp(two_pi_of<scalar_t>() * r_fracs_shifted, -point_limit, point_limit).contiguous();
        auto r_in_t = r_in.transpose(0, 1).contiguous();

        auto q_real = latent_charge[f].reshape({-1}).contiguous();
        TORCH_CHECK(q_real.numel() == nloc, "latent_charge per frame must contain nloc values");
        auto charge_f = at::complex(q_real, at::zeros_like(q_real)).contiguous();

        int ier_set_t1 = finufft_setpts<scalar_t>(
            plan_t1,
            static_cast<int64_t>(nloc),
            r_in_t[0].template data_ptr<scalar_t>(),
            r_in_t[1].template data_ptr<scalar_t>(),
            r_in_t[2].template data_ptr<scalar_t>()
        );
        TORCH_CHECK(ier_set_t1 == 0, "cufinufftf_setpts(t1,bwd) failed.");

        int dim_x = 2 * nkx + 1;
        int dim_y = 2 * nky + 1;
        int dim_z = 2 * nkz + 1;
        int total_grid = dim_x * dim_y * dim_z;

        auto grid_out = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        int ier_exec_t1 = finufft_execute<scalar_t>(
            plan_t1,
            reinterpret_cast<scalar_t*>(charge_f.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(grid_out.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t1 == 0, "cufinufftf_execute(t1,bwd) failed.");

        auto conv_grid = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        auto sum_base_rho = at::zeros({1}, coord.options());
        auto sum_base = at::zeros({1}, coord.options());
        auto sum_dbw_rho = at::zeros({num_bandwidth}, coord.options());
        auto sum_dbw = at::zeros({num_bandwidth}, coord.options());

        launch_sog_backward_energy_kernel(
            nkx,
            nky,
            nkz,
            num_bandwidth,
            reinterpret_cast<const scalar_t*>(grid_out.template data_ptr<complex_t>()),
            amp.template data_ptr<scalar_t>(),
            bw_sq.template data_ptr<scalar_t>(),
            cells_inv[f].template data_ptr<scalar_t>(),
            reinterpret_cast<scalar_t*>(conv_grid.template data_ptr<complex_t>()),
            sum_base_rho.template data_ptr<scalar_t>(),
            sum_base.template data_ptr<scalar_t>(),
            sum_dbw_rho.template data_ptr<scalar_t>(),
            sum_dbw.template data_ptr<scalar_t>(),
            stream
        );

        cufinufft_plan plan_t2 = get_or_create_plan<scalar_t>(nkx, nky, nkz, 2);
        int ier_set_t2 = finufft_setpts<scalar_t>(
            plan_t2,
            static_cast<int64_t>(nloc),
            r_in_t[0].template data_ptr<scalar_t>(),
            r_in_t[1].template data_ptr<scalar_t>(),
            r_in_t[2].template data_ptr<scalar_t>()
        );
        TORCH_CHECK(ier_set_t2 == 0, "cufinufftf_setpts(t2,bwd) failed.");

        auto phi = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        int ier_exec_t2 = finufft_execute<scalar_t>(
            plan_t2,
            reinterpret_cast<scalar_t*>(phi.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(conv_grid.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t2 == 0, "cufinufftf_execute(t2,bwd) failed.");

        auto volume_f = volumes[f];
        auto grad_frame = grad_corr[f][0];
        auto q_sq_sum = at::sum(q_real * q_real);

        auto pref_force = 1.0 / volume_f;
        auto grad_q = at::real(phi) * pref_force;
        if (remove_self_interaction) {
            auto diag_sum = amp.reshape({-1})[0] * sum_base[0];
            grad_q = grad_q - q_real * (diag_sum * pref_force);
        }
        grad_q = grad_q * grad_frame;
        grad_latent[f].reshape({-1}).copy_(grad_q);

        auto inv_2v = 1.0 / (2.0 * volume_f);
        auto grad_amp_frame = sum_base_rho * inv_2v;
        auto grad_bw_frame = sum_dbw_rho * inv_2v;
        if (remove_self_interaction) {
            grad_amp_frame = grad_amp_frame - q_sq_sum * (sum_base * inv_2v);
            grad_bw_frame = grad_bw_frame - q_sq_sum * (sum_dbw * inv_2v);
        }
        grad_amp = grad_amp + grad_amp_frame * grad_frame;
        grad_bandwidth = grad_bandwidth + grad_bw_frame * grad_frame;
    }

    std::vector<at::Tensor> res;
    res.push_back(grad_latent);
    res.push_back(grad_amp);
    res.push_back(grad_bandwidth);
    return res;
}

template <typename scalar_t>
static std::vector<at::Tensor> nufft_sog_frame_correction_backward_bundle_impl(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr,
    const at::Tensor& grad_force,
    const at::Tensor& force_forward,
    bool need_latent_grad
) {
    static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>);
    using complex_t = c10::complex<scalar_t>;

    TORCH_CHECK(coord.is_cuda(), "coord must be a CUDA tensor");
    TORCH_CHECK(latent_charge.is_cuda(), "latent_charge must be a CUDA tensor");
    TORCH_CHECK(box.is_cuda(), "box must be a CUDA tensor");
    TORCH_CHECK(amp.is_cuda(), "amp must be a CUDA tensor");
    TORCH_CHECK(bandwidth.is_cuda(), "bandwidth must be a CUDA tensor");
    TORCH_CHECK(grad_corr.is_cuda(), "grad_corr must be a CUDA tensor");
    TORCH_CHECK(grad_force.is_cuda(), "grad_force must be a CUDA tensor");
    TORCH_CHECK(force_forward.is_cuda(), "force_forward must be a CUDA tensor");
    constexpr at::ScalarType expected_dtype = std::is_same_v<scalar_t, float> ? at::kFloat : at::kDouble;
    TORCH_CHECK(coord.scalar_type() == expected_dtype, "coord dtype mismatch with kernel type");
    TORCH_CHECK(latent_charge.scalar_type() == expected_dtype, "latent_charge dtype mismatch with kernel type");
    TORCH_CHECK(box.scalar_type() == expected_dtype, "box dtype mismatch with kernel type");
    TORCH_CHECK(amp.scalar_type() == expected_dtype, "amp dtype mismatch with kernel type");
    TORCH_CHECK(bandwidth.scalar_type() == expected_dtype, "bandwidth dtype mismatch with kernel type");
    TORCH_CHECK(grad_corr.scalar_type() == expected_dtype, "grad_corr dtype mismatch with kernel type");
    TORCH_CHECK(grad_force.scalar_type() == expected_dtype, "grad_force dtype mismatch with kernel type");
    TORCH_CHECK(force_forward.scalar_type() == expected_dtype, "force_forward dtype mismatch with kernel type");

    int nf = coord.size(0);
    int nloc = coord.size(1);
    TORCH_CHECK(nk_cpu.size() == static_cast<size_t>(nf * 3), "nk_cpu length must be nf*3");
    TORCH_CHECK(amp.numel() == 1, "amp must contain exactly one scalar value");
    TORCH_CHECK(grad_corr.dim() == 2 && grad_corr.size(0) == nf && grad_corr.size(1) == 1,
                "grad_corr must be [nf, 1]");
    TORCH_CHECK(grad_force.dim() == 3 && grad_force.size(0) == nf && grad_force.size(1) == nloc && grad_force.size(2) == 3,
                "grad_force must be [nf, nloc, 3]");
    if (need_latent_grad) {
        TORCH_CHECK(force_forward.dim() == 3 && force_forward.size(0) == nf && force_forward.size(1) == nloc && force_forward.size(2) == 3,
                    "force_forward must be [nf, nloc, 3]");
    }

    auto grad_latent = at::zeros_like(latent_charge);
    auto grad_amp = at::zeros_like(amp);
    auto grad_bandwidth = at::zeros_like(bandwidth);

    // Reuse existing energy backward to preserve energy-path correctness.
    auto grad_corr_abs_sum = at::sum(at::abs(grad_corr));
    bool has_energy_contrib = grad_corr_abs_sum.item<double>() > 0.0;
    if (has_energy_contrib) {
        auto base = nufft_sog_frame_correction_backward_energy_impl<scalar_t>(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            remove_self_interaction,
            grad_corr
        );
        grad_latent = base[0];
        grad_amp = base[1];
        grad_bandwidth = base[2];
    }

    auto grad_force_abs_sum = at::sum(at::abs(grad_force));
    bool has_force_contrib = grad_force_abs_sum.item<double>() > 0.0;
    if (!has_force_contrib) {
        std::vector<at::Tensor> res;
        res.push_back(grad_latent);
        res.push_back(grad_amp);
        res.push_back(grad_bandwidth);
        return res;
    }

    auto bw_sq = at::square(bandwidth);
    int num_bandwidth = static_cast<int>(bw_sq.numel());

    auto& finufft = CuFinufftLib::getInstance();
    TORCH_CHECK(finufft.handle != nullptr, "Could not load libcufinufft.so");

    auto volumes = at::linalg_det(box);
    auto cells_inv = at::linalg_inv(box);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    for (int f = 0; f < nf; ++f) {
        int nkx = nk_cpu[f * 3 + 0];
        int nky = nk_cpu[f * 3 + 1];
        int nkz = nk_cpu[f * 3 + 2];
        TORCH_CHECK(nkx > 0 && nky > 0 && nkz > 0, "nk_cpu entries must be positive");

        cufinufft_plan plan_t1 = get_or_create_plan<scalar_t>(nkx, nky, nkz, 1);

        auto r_fracs_f = at::einsum("ni,ij->nj", {coord[f], cells_inv[f]});
        auto r_fracs_shifted = at::remainder(
            r_fracs_f + static_cast<scalar_t>(0.5),
            static_cast<scalar_t>(1.0)
        ) - static_cast<scalar_t>(0.5);
        const scalar_t point_limit = point_limit_of<scalar_t>();
        auto r_in = at::clamp(two_pi_of<scalar_t>() * r_fracs_shifted, -point_limit, point_limit).contiguous();
        auto r_in_t = r_in.transpose(0, 1).contiguous();

        auto q_real = latent_charge[f].reshape({-1}).contiguous();
        TORCH_CHECK(q_real.numel() == nloc, "latent_charge per frame must contain nloc values");

        int ier_set_t1 = finufft_setpts<scalar_t>(
            plan_t1,
            static_cast<int64_t>(nloc),
            r_in_t[0].template data_ptr<scalar_t>(),
            r_in_t[1].template data_ptr<scalar_t>(),
            r_in_t[2].template data_ptr<scalar_t>()
        );
        TORCH_CHECK(ier_set_t1 == 0, "cufinufftf_setpts(t1,bwd_force) failed.");

        int dim_x = 2 * nkx + 1;
        int dim_y = 2 * nky + 1;
        int dim_z = 2 * nkz + 1;
        int total_grid = dim_x * dim_y * dim_z;

        auto charge_q = at::complex(q_real, at::zeros_like(q_real)).contiguous();
        auto rho_grid = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        int ier_exec_t1_q = finufft_execute<scalar_t>(
            plan_t1,
            reinterpret_cast<scalar_t*>(charge_q.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(rho_grid.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t1_q == 0, "cufinufftf_execute(t1,q,bwd_force) failed.");

        cufinufft_plan plan_t2 = nullptr;
        if (need_latent_grad) {
            plan_t2 = get_or_create_plan<scalar_t>(nkx, nky, nkz, 2);
            int ier_set_t2 = finufft_setpts<scalar_t>(
                plan_t2,
                static_cast<int64_t>(nloc),
                r_in_t[0].template data_ptr<scalar_t>(),
                r_in_t[1].template data_ptr<scalar_t>(),
                r_in_t[2].template data_ptr<scalar_t>()
            );
            TORCH_CHECK(ier_set_t2 == 0, "cufinufftf_setpts(t2,bwd_force) failed.");
        }

        auto grad_force_f = grad_force[f].contiguous();
        auto gx = grad_force_f.select(1, 0).contiguous();
        auto gy = grad_force_f.select(1, 1).contiguous();
        auto gz = grad_force_f.select(1, 2).contiguous();

        auto coeff_sx = at::complex(q_real * gx, at::zeros_like(q_real)).contiguous();
        auto coeff_sy = at::complex(q_real * gy, at::zeros_like(q_real)).contiguous();
        auto coeff_sz = at::complex(q_real * gz, at::zeros_like(q_real)).contiguous();

        auto sminus_x = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        auto sminus_y = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        auto sminus_z = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));

        int ier_exec_t1_sx = finufft_execute<scalar_t>(
            plan_t1,
            reinterpret_cast<scalar_t*>(coeff_sx.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(sminus_x.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t1_sx == 0, "cufinufftf_execute(t1,sx,bwd_force) failed.");
        int ier_exec_t1_sy = finufft_execute<scalar_t>(
            plan_t1,
            reinterpret_cast<scalar_t*>(coeff_sy.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(sminus_y.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t1_sy == 0, "cufinufftf_execute(t1,sy,bwd_force) failed.");
        int ier_exec_t1_sz = finufft_execute<scalar_t>(
            plan_t1,
            reinterpret_cast<scalar_t*>(coeff_sz.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(sminus_z.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t1_sz == 0, "cufinufftf_execute(t1,sz,bwd_force) failed.");

        auto sum_base_force = at::zeros({1}, coord.options());
        auto sum_dbw_force = at::zeros({num_bandwidth}, coord.options());
        auto indirect_coeff_x = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        auto indirect_coeff_y = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        auto indirect_coeff_z = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));

        auto volume_f = volumes[f];
        scalar_t inv_volume = static_cast<scalar_t>(1.0) / volume_f.item<scalar_t>();
        launch_sog_backward_force_prepare_kernel(
            nkx,
            nky,
            nkz,
            num_bandwidth,
            reinterpret_cast<const scalar_t*>(rho_grid.template data_ptr<complex_t>()),
            reinterpret_cast<const scalar_t*>(sminus_x.template data_ptr<complex_t>()),
            reinterpret_cast<const scalar_t*>(sminus_y.template data_ptr<complex_t>()),
            reinterpret_cast<const scalar_t*>(sminus_z.template data_ptr<complex_t>()),
            amp.template data_ptr<scalar_t>(),
            bw_sq.template data_ptr<scalar_t>(),
            cells_inv[f].template data_ptr<scalar_t>(),
            inv_volume,
            sum_base_force.template data_ptr<scalar_t>(),
            sum_dbw_force.template data_ptr<scalar_t>(),
            reinterpret_cast<scalar_t*>(indirect_coeff_x.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(indirect_coeff_y.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(indirect_coeff_z.template data_ptr<complex_t>()),
            stream
        );

        if (need_latent_grad) {
            auto indirect_x = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto indirect_y = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto indirect_z = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            int ier_exec_t2_ix = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(indirect_x.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(indirect_coeff_x.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_ix == 0, "cufinufftf_execute(t2,indirect_x,bwd_force) failed.");
            int ier_exec_t2_iy = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(indirect_y.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(indirect_coeff_y.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_iy == 0, "cufinufftf_execute(t2,indirect_y,bwd_force) failed.");
            int ier_exec_t2_iz = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(indirect_z.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(indirect_coeff_z.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_iz == 0, "cufinufftf_execute(t2,indirect_z,bwd_force) failed.");

            auto force_f = force_forward[f].contiguous();
            auto force_dot =
                gx * force_f.select(1, 0).contiguous() +
                gy * force_f.select(1, 1).contiguous() +
                gz * force_f.select(1, 2).contiguous();
            auto explicit_latent = at::where(
                at::abs(q_real) > static_cast<scalar_t>(1e-14),
                force_dot / q_real,
                at::zeros_like(force_dot)
            );
            auto indirect_latent = -(
                at::real(indirect_x) +
                at::real(indirect_y) +
                at::real(indirect_z)
            ) * inv_volume;
            auto grad_q_force = explicit_latent + indirect_latent;

            grad_latent[f].reshape({-1}).add_(grad_q_force);
        }
        grad_amp = grad_amp + sum_base_force;
        grad_bandwidth = grad_bandwidth + sum_dbw_force;
    }

    std::vector<at::Tensor> res;
    res.push_back(grad_latent);
    res.push_back(grad_amp);
    res.push_back(grad_bandwidth);
    return res;
}

std::vector<at::Tensor> nufft_sog_frame_correction_bundle(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    bool need_force,
    bool need_virial
) {
    check_sog_dtype_supported(coord, "coord");
    check_sog_dtype_supported(latent_charge, "latent_charge");
    check_sog_dtype_supported(box, "box");
    check_sog_dtype_supported(amp, "amp");
    check_sog_dtype_supported(bandwidth, "bandwidth");
    check_sog_same_dtype(coord, latent_charge, box, amp, bandwidth);

    if (coord.scalar_type() == at::kDouble) {
        return nufft_sog_frame_correction_bundle_impl<double>(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            remove_self_interaction,
            need_force,
            need_virial
        );
    }

    return nufft_sog_frame_correction_bundle_impl<float>(
        coord,
        latent_charge,
        box,
        amp,
        bandwidth,
        nk_cpu,
        remove_self_interaction,
        need_force,
        need_virial
    );
}

std::vector<at::Tensor> nufft_sog_frame_correction_backward_energy(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr
) {
    check_sog_dtype_supported(coord, "coord");
    check_sog_dtype_supported(latent_charge, "latent_charge");
    check_sog_dtype_supported(box, "box");
    check_sog_dtype_supported(amp, "amp");
    check_sog_dtype_supported(bandwidth, "bandwidth");
    check_sog_dtype_supported(grad_corr, "grad_corr");
    check_sog_same_dtype(
        coord,
        latent_charge,
        box,
        amp,
        bandwidth,
        &grad_corr,
        nullptr,
        nullptr
    );

    if (coord.scalar_type() == at::kDouble) {
        return nufft_sog_frame_correction_backward_energy_impl<double>(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            remove_self_interaction,
            grad_corr
        );
    }

    return nufft_sog_frame_correction_backward_energy_impl<float>(
        coord,
        latent_charge,
        box,
        amp,
        bandwidth,
        nk_cpu,
        remove_self_interaction,
        grad_corr
    );
}

std::vector<at::Tensor> nufft_sog_frame_correction_backward_bundle(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& amp,
    const at::Tensor& bandwidth,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr,
    const at::Tensor& grad_force,
    const at::Tensor& force_forward,
    bool need_latent_grad
) {
    check_sog_dtype_supported(coord, "coord");
    check_sog_dtype_supported(latent_charge, "latent_charge");
    check_sog_dtype_supported(box, "box");
    check_sog_dtype_supported(amp, "amp");
    check_sog_dtype_supported(bandwidth, "bandwidth");
    check_sog_dtype_supported(grad_corr, "grad_corr");
    check_sog_dtype_supported(grad_force, "grad_force");
    check_sog_dtype_supported(force_forward, "force_forward");
    check_sog_same_dtype(
        coord,
        latent_charge,
        box,
        amp,
        bandwidth,
        &grad_corr,
        &grad_force,
        &force_forward
    );

    if (coord.scalar_type() == at::kDouble) {
        return nufft_sog_frame_correction_backward_bundle_impl<double>(
            coord,
            latent_charge,
            box,
            amp,
            bandwidth,
            nk_cpu,
            remove_self_interaction,
            grad_corr,
            grad_force,
            force_forward,
            need_latent_grad
        );
    }

    return nufft_sog_frame_correction_backward_bundle_impl<float>(
        coord,
        latent_charge,
        box,
        amp,
        bandwidth,
        nk_cpu,
        remove_self_interaction,
        grad_corr,
        grad_force,
        force_forward,
        need_latent_grad
    );
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
    m.def("nufft_sog_frame_correction_bundle(Tensor coord, Tensor latent_charge, Tensor box, Tensor amp, Tensor bandwidth, int[] nk_cpu, bool remove_self_interaction, bool need_force, bool need_virial) -> Tensor[]");
    m.impl("nufft_sog_frame_correction_bundle", torch::kCUDA, &nufft_sog_frame_correction_bundle);
    m.def("nufft_sog_frame_correction_backward_energy(Tensor coord, Tensor latent_charge, Tensor box, Tensor amp, Tensor bandwidth, int[] nk_cpu, bool remove_self_interaction, Tensor grad_corr) -> Tensor[]");
    m.impl("nufft_sog_frame_correction_backward_energy", torch::kCUDA, &nufft_sog_frame_correction_backward_energy);
    m.def("nufft_sog_frame_correction_backward_bundle(Tensor coord, Tensor latent_charge, Tensor box, Tensor amp, Tensor bandwidth, int[] nk_cpu, bool remove_self_interaction, Tensor grad_corr, Tensor grad_force, Tensor force_forward, bool need_latent_grad) -> Tensor[]");
    m.impl("nufft_sog_frame_correction_backward_bundle", torch::kCUDA, &nufft_sog_frame_correction_backward_bundle);
}
