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

void launch_les_modulate_kernel(
    int nkx,
    int nky,
    int nkz,
    const double* coeff_ptr,
    double sigma_sq,
    const double* cell_inv_ptr,
    double* out_corr_ptr,
    double* out_kfac_sum_ptr,
    cudaStream_t stream
);

void launch_les_modulate_kernel(
    int nkx,
    int nky,
    int nkz,
    const float* coeff_ptr,
    float sigma_sq,
    const float* cell_inv_ptr,
    float* out_corr_ptr,
    float* out_kfac_sum_ptr,
    cudaStream_t stream
);

void launch_les_build_gradconv_kernel(
    int nkx,
    int nky,
    int nkz,
    const double* coeff_ptr,
    double sigma_sq,
    const double* cell_inv_ptr,
    double* gradconv_x_ptr,
    double* gradconv_y_ptr,
    double* gradconv_z_ptr,
    cudaStream_t stream
);

void launch_les_build_gradconv_kernel(
    int nkx,
    int nky,
    int nkz,
    const float* coeff_ptr,
    float sigma_sq,
    const float* cell_inv_ptr,
    float* gradconv_x_ptr,
    float* gradconv_y_ptr,
    float* gradconv_z_ptr,
    cudaStream_t stream
);

void launch_les_backward_energy_kernel(
    int nkx,
    int nky,
    int nkz,
    const double* coeff_ptr,
    double sigma,
    double sigma_sq,
    const double* cell_inv_ptr,
    double* conv_grid_ptr,
    double* sum_kfac_ptr,
    double* sum_dsigma_rho_ptr,
    double* sum_dsigma_ptr,
    cudaStream_t stream
);

void launch_les_backward_energy_kernel(
    int nkx,
    int nky,
    int nkz,
    const float* coeff_ptr,
    float sigma,
    float sigma_sq,
    const float* cell_inv_ptr,
    float* conv_grid_ptr,
    float* sum_kfac_ptr,
    float* sum_dsigma_rho_ptr,
    float* sum_dsigma_ptr,
    cudaStream_t stream
);

struct LESPlanKey {
    int nkx, nky, nkz;
    int type;
    int precision;

    bool operator==(const LESPlanKey& other) const {
        return nkx == other.nkx && nky == other.nky && nkz == other.nkz &&
               type == other.type && precision == other.precision;
    }
};

namespace std {
template <>
struct hash<LESPlanKey> {
    size_t operator()(const LESPlanKey& k) const {
        return (hash<int>()(k.nkx) ^ (hash<int>()(k.nky) << 1)) ^
               (hash<int>()(k.nkz) << 2) ^ (hash<int>()(k.type) << 3) ^
               (hash<int>()(k.precision) << 4);
    }
};
}  // namespace std

static std::unordered_map<LESPlanKey, cufinufft_plan> g_les_plan_cache;
static std::mutex g_les_plan_mutex;

template <typename scalar_t>
cufinufft_plan get_or_create_les_plan(int nkx, int nky, int nkz, int type) {
    constexpr int precision = std::is_same_v<scalar_t, float> ? 0 : 1;
    LESPlanKey key = {nkx, nky, nkz, type, precision};
    std::lock_guard<std::mutex> lock(g_les_plan_mutex);
    if (g_les_plan_cache.find(key) != g_les_plan_cache.end()) {
        return g_les_plan_cache[key];
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
    int64_t nmodes[3] = {(2 * nkx + 1), (2 * nky + 1), (2 * nkz + 1)};

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

    g_les_plan_cache[key] = plan;
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

std::vector<at::Tensor> nufft_les_frame_correction_bundle(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    bool need_force,
    bool need_virial
);

std::vector<at::Tensor> nufft_les_frame_correction_backward_energy(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr
);

namespace {

void check_les_dtype_supported(const at::Tensor& t, const char* name) {
    TORCH_CHECK(
        t.scalar_type() == at::kDouble || t.scalar_type() == at::kFloat,
        name,
        " must be float32 or float64"
    );
}

void check_les_same_dtype(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
    const at::Tensor* grad_corr = nullptr
) {
    auto dtype = coord.scalar_type();
    TORCH_CHECK(latent_charge.scalar_type() == dtype, "latent_charge dtype must match coord dtype");
    TORCH_CHECK(box.scalar_type() == dtype, "box dtype must match coord dtype");
    TORCH_CHECK(sigma.scalar_type() == dtype, "sigma dtype must match coord dtype");
    if (grad_corr != nullptr) {
        TORCH_CHECK(grad_corr->scalar_type() == dtype, "grad_corr dtype must match coord dtype");
    }
}

}  // namespace

template <typename scalar_t>
static std::vector<at::Tensor> nufft_les_frame_correction_bundle_impl(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
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
    TORCH_CHECK(sigma.is_cuda(), "sigma must be a CUDA tensor");
    constexpr at::ScalarType expected_dtype = std::is_same_v<scalar_t, float> ? at::kFloat : at::kDouble;
    TORCH_CHECK(coord.scalar_type() == expected_dtype, "coord dtype mismatch with kernel type");
    TORCH_CHECK(latent_charge.scalar_type() == expected_dtype, "latent_charge dtype mismatch with kernel type");
    TORCH_CHECK(box.scalar_type() == expected_dtype, "box dtype mismatch with kernel type");
    TORCH_CHECK(sigma.scalar_type() == expected_dtype, "sigma dtype mismatch with kernel type");
    TORCH_CHECK(coord.dim() == 3 && coord.size(2) == 3,
                "coord must be [nf, nloc, 3]");
    TORCH_CHECK(box.dim() == 3 && box.size(1) == 3 && box.size(2) == 3,
                "box must be [nf, 3, 3]");

    int nf = coord.size(0);
    int nloc = coord.size(1);
    TORCH_CHECK(nk_cpu.size() == static_cast<size_t>(nf * 3),
                "nk_cpu length must be nf*3");
    TORCH_CHECK(sigma.numel() == 1, "sigma must contain exactly one scalar value");

    const scalar_t sigma_val = sigma.reshape({-1})[0].item<scalar_t>();
    const scalar_t sigma_sq = sigma_val * sigma_val;

    auto corr_out = at::zeros({nf, 1}, coord.options());
    auto kfac_sum_out = at::zeros({nf, 1}, coord.options());
    auto force_out =
        need_force ? at::zeros({nf, nloc, 3}, coord.options()) : at::Tensor();
    auto virial_out =
        need_virial ? at::zeros({nf, nloc, 1, 9}, coord.options()) : at::Tensor();

    auto& finufft = CuFinufftLib::getInstance();
    TORCH_CHECK(finufft.handle != nullptr, "Could not load libcufinufft.so");

    auto volumes = at::linalg_det(box);
    auto cells_inv = at::linalg_inv(box);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    for (int f = 0; f < nf; ++f) {
        int nkx = nk_cpu[f * 3 + 0];
        int nky = nk_cpu[f * 3 + 1];
        int nkz = nk_cpu[f * 3 + 2];
        TORCH_CHECK(nkx > 0 && nky > 0 && nkz > 0,
                    "nk_cpu entries must be positive");

        cufinufft_plan plan = get_or_create_les_plan<scalar_t>(nkx, nky, nkz, 1);

        auto r_fracs_f = at::einsum("ni,ij->nj", {coord[f], cells_inv[f]});
        auto r_fracs_shifted = at::remainder(
            r_fracs_f + static_cast<scalar_t>(0.5),
            static_cast<scalar_t>(1.0)
        ) - static_cast<scalar_t>(0.5);
        const scalar_t point_limit = point_limit_of<scalar_t>();
        auto r_in = at::clamp(two_pi_of<scalar_t>() * r_fracs_shifted, -point_limit,
                              point_limit)
                        .contiguous();

        auto q_real = latent_charge[f].reshape({-1}).contiguous();
        TORCH_CHECK(q_real.numel() == nloc,
                    "latent_charge per frame must contain nloc values");
        auto charge_f = at::complex(q_real, at::zeros_like(q_real)).contiguous();

        auto r_in_t = r_in.transpose(0, 1).contiguous();

        int ier_set = finufft_setpts<scalar_t>(
            plan,
            static_cast<int64_t>(nloc),
            r_in_t[0].template data_ptr<scalar_t>(),
            r_in_t[1].template data_ptr<scalar_t>(),
            r_in_t[2].template data_ptr<scalar_t>()
        );
        TORCH_CHECK(ier_set == 0, "cufinufft_setpts failed.");

        int dim_x = 2 * nkx + 1;
        int dim_y = 2 * nky + 1;
        int dim_z = 2 * nkz + 1;
        int total_grid = dim_x * dim_y * dim_z;
        auto grid_out =
            at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));

        int ier_exec = finufft_execute<scalar_t>(
            plan,
            reinterpret_cast<scalar_t*>(charge_f.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(grid_out.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec == 0, "cufinufft_execute failed.");

        launch_les_modulate_kernel(
            nkx, nky, nkz,
            reinterpret_cast<const scalar_t*>(
                grid_out.template data_ptr<complex_t>()),
            sigma_sq,
            cells_inv[f].template data_ptr<scalar_t>(),
            corr_out[f].template data_ptr<scalar_t>(),
            kfac_sum_out[f].template data_ptr<scalar_t>(),
            stream);

        auto volume_f = volumes[f];
        auto prefac = two_pi_of<scalar_t>() / volume_f;
        corr_out[f] = corr_out[f] * prefac;

        if (remove_self_interaction) {
            auto q_sq_sum = at::sum(q_real * q_real);
            corr_out[f] = corr_out[f] - q_sq_sum * (kfac_sum_out[f] * prefac);
        }

        if (need_force || need_virial) {
            auto grad_conv_x = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_conv_y = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_conv_z = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));

            launch_les_build_gradconv_kernel(
                nkx,
                nky,
                nkz,
                reinterpret_cast<const scalar_t*>(grid_out.template data_ptr<complex_t>()),
                sigma_sq,
                cells_inv[f].template data_ptr<scalar_t>(),
                reinterpret_cast<scalar_t*>(grad_conv_x.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_y.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_z.template data_ptr<complex_t>()),
                stream
            );

            cufinufft_plan plan_t2 = get_or_create_les_plan<scalar_t>(nkx, nky, nkz, 2);
            int ier_set_t2 = finufft_setpts<scalar_t>(
                plan_t2,
                static_cast<int64_t>(nloc),
                r_in_t[0].template data_ptr<scalar_t>(),
                r_in_t[1].template data_ptr<scalar_t>(),
                r_in_t[2].template data_ptr<scalar_t>()
            );
            TORCH_CHECK(ier_set_t2 == 0, "cufinufft_setpts(type2) failed.");

            auto grad_field_x = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_field_y = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
            auto grad_field_z = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));

            int ier_exec_t2_x = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(grad_field_x.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_x.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_x == 0, "cufinufft_execute(type2,x) failed.");

            int ier_exec_t2_y = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(grad_field_y.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_y.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_y == 0, "cufinufft_execute(type2,y) failed.");

            int ier_exec_t2_z = finufft_execute<scalar_t>(
                plan_t2,
                reinterpret_cast<scalar_t*>(grad_field_z.template data_ptr<complex_t>()),
                reinterpret_cast<scalar_t*>(grad_conv_z.template data_ptr<complex_t>())
            );
            TORCH_CHECK(ier_exec_t2_z == 0, "cufinufft_execute(type2,z) failed.");

            auto force_prefac = (two_pi_of<scalar_t>() * static_cast<scalar_t>(2.0)) / volume_f;
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
static std::vector<at::Tensor> nufft_les_frame_correction_backward_energy_impl(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr
) {
    static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>);
    using complex_t = c10::complex<scalar_t>;

    TORCH_CHECK(coord.is_cuda(), "coord must be a CUDA tensor");
    TORCH_CHECK(latent_charge.is_cuda(), "latent_charge must be a CUDA tensor");
    TORCH_CHECK(box.is_cuda(), "box must be a CUDA tensor");
    TORCH_CHECK(sigma.is_cuda(), "sigma must be a CUDA tensor");
    TORCH_CHECK(grad_corr.is_cuda(), "grad_corr must be a CUDA tensor");
    constexpr at::ScalarType expected_dtype = std::is_same_v<scalar_t, float> ? at::kFloat : at::kDouble;
    TORCH_CHECK(coord.scalar_type() == expected_dtype, "coord dtype mismatch with kernel type");
    TORCH_CHECK(latent_charge.scalar_type() == expected_dtype, "latent_charge dtype mismatch with kernel type");
    TORCH_CHECK(box.scalar_type() == expected_dtype, "box dtype mismatch with kernel type");
    TORCH_CHECK(sigma.scalar_type() == expected_dtype, "sigma dtype mismatch with kernel type");
    TORCH_CHECK(grad_corr.scalar_type() == expected_dtype, "grad_corr dtype mismatch with kernel type");

    int nf = coord.size(0);
    int nloc = coord.size(1);
    TORCH_CHECK(nk_cpu.size() == static_cast<size_t>(nf * 3), "nk_cpu length must be nf*3");
    TORCH_CHECK(sigma.numel() == 1, "sigma must contain exactly one scalar value");
    TORCH_CHECK(grad_corr.dim() == 2 && grad_corr.size(0) == nf && grad_corr.size(1) == 1,
                "grad_corr must be [nf, 1]");

    auto grad_latent = at::zeros_like(latent_charge);
    auto grad_sigma = at::zeros_like(sigma);

    auto& finufft = CuFinufftLib::getInstance();
    TORCH_CHECK(finufft.handle != nullptr, "Could not load libcufinufft.so");

    auto volumes = at::linalg_det(box);
    auto cells_inv = at::linalg_inv(box);
    scalar_t sigma_val = sigma.reshape({-1})[0].item<scalar_t>();
    scalar_t sigma_sq = sigma_val * sigma_val;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    for (int f = 0; f < nf; ++f) {
        int nkx = nk_cpu[f * 3 + 0];
        int nky = nk_cpu[f * 3 + 1];
        int nkz = nk_cpu[f * 3 + 2];
        TORCH_CHECK(nkx > 0 && nky > 0 && nkz > 0, "nk_cpu entries must be positive");

        cufinufft_plan plan_t1 = get_or_create_les_plan<scalar_t>(nkx, nky, nkz, 1);

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
        TORCH_CHECK(ier_set_t1 == 0, "cufinufft_setpts(t1,bwd) failed.");

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
        TORCH_CHECK(ier_exec_t1 == 0, "cufinufft_execute(t1,bwd) failed.");

        auto conv_grid = at::zeros({total_grid}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        auto sum_kfac = at::zeros({1}, coord.options());
        auto sum_dsigma_rho = at::zeros({1}, coord.options());
        auto sum_dsigma = at::zeros({1}, coord.options());

        launch_les_backward_energy_kernel(
            nkx,
            nky,
            nkz,
            reinterpret_cast<const scalar_t*>(grid_out.template data_ptr<complex_t>()),
            sigma_val,
            sigma_sq,
            cells_inv[f].template data_ptr<scalar_t>(),
            reinterpret_cast<scalar_t*>(conv_grid.template data_ptr<complex_t>()),
            sum_kfac.template data_ptr<scalar_t>(),
            sum_dsigma_rho.template data_ptr<scalar_t>(),
            sum_dsigma.template data_ptr<scalar_t>(),
            stream
        );

        cufinufft_plan plan_t2 = get_or_create_les_plan<scalar_t>(nkx, nky, nkz, 2);
        int ier_set_t2 = finufft_setpts<scalar_t>(
            plan_t2,
            static_cast<int64_t>(nloc),
            r_in_t[0].template data_ptr<scalar_t>(),
            r_in_t[1].template data_ptr<scalar_t>(),
            r_in_t[2].template data_ptr<scalar_t>()
        );
        TORCH_CHECK(ier_set_t2 == 0, "cufinufft_setpts(t2,bwd) failed.");

        auto phi = at::zeros({nloc}, coord.options().dtype(complex_dtype_of<scalar_t>()));
        int ier_exec_t2 = finufft_execute<scalar_t>(
            plan_t2,
            reinterpret_cast<scalar_t*>(phi.template data_ptr<complex_t>()),
            reinterpret_cast<scalar_t*>(conv_grid.template data_ptr<complex_t>())
        );
        TORCH_CHECK(ier_exec_t2 == 0, "cufinufft_execute(t2,bwd) failed.");

        auto volume_f = volumes[f];
        auto grad_frame = grad_corr[f][0];
        auto q_sq_sum = at::sum(q_real * q_real);

        auto pref_latent = (two_pi_of<scalar_t>() * static_cast<scalar_t>(2.0)) / volume_f;
        auto grad_q = at::real(phi) * pref_latent;
        if (remove_self_interaction) {
            grad_q = grad_q - q_real * (sum_kfac[0] * pref_latent);
        }
        grad_q = grad_q * grad_frame;
        grad_latent[f].reshape({-1}).copy_(grad_q);

        auto pref_sigma = two_pi_of<scalar_t>() / volume_f;
        auto grad_sigma_frame = sum_dsigma_rho * pref_sigma;
        if (remove_self_interaction) {
            grad_sigma_frame = grad_sigma_frame - q_sq_sum * (sum_dsigma * pref_sigma);
        }
        grad_sigma = grad_sigma + grad_sigma_frame * grad_frame;
    }

    std::vector<at::Tensor> res;
    res.push_back(grad_latent);
    res.push_back(grad_sigma);
    return res;
}

std::vector<at::Tensor> nufft_les_frame_correction_bundle(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    bool need_force,
    bool need_virial
) {
    check_les_dtype_supported(coord, "coord");
    check_les_dtype_supported(latent_charge, "latent_charge");
    check_les_dtype_supported(box, "box");
    check_les_dtype_supported(sigma, "sigma");
    check_les_same_dtype(coord, latent_charge, box, sigma);

    if (coord.scalar_type() == at::kDouble) {
        return nufft_les_frame_correction_bundle_impl<double>(
            coord,
            latent_charge,
            box,
            sigma,
            nk_cpu,
            remove_self_interaction,
            need_force,
            need_virial
        );
    }

    return nufft_les_frame_correction_bundle_impl<float>(
        coord,
        latent_charge,
        box,
        sigma,
        nk_cpu,
        remove_self_interaction,
        need_force,
        need_virial
    );
}

std::vector<at::Tensor> nufft_les_frame_correction_backward_energy(
    const at::Tensor& coord,
    const at::Tensor& latent_charge,
    const at::Tensor& box,
    const at::Tensor& sigma,
    const std::vector<int64_t>& nk_cpu,
    bool remove_self_interaction,
    const at::Tensor& grad_corr
) {
    check_les_dtype_supported(coord, "coord");
    check_les_dtype_supported(latent_charge, "latent_charge");
    check_les_dtype_supported(box, "box");
    check_les_dtype_supported(sigma, "sigma");
    check_les_dtype_supported(grad_corr, "grad_corr");
    check_les_same_dtype(coord, latent_charge, box, sigma, &grad_corr);

    if (coord.scalar_type() == at::kDouble) {
        return nufft_les_frame_correction_backward_energy_impl<double>(
            coord,
            latent_charge,
            box,
            sigma,
            nk_cpu,
            remove_self_interaction,
            grad_corr
        );
    }

    return nufft_les_frame_correction_backward_energy_impl<float>(
        coord,
        latent_charge,
        box,
        sigma,
        nk_cpu,
        remove_self_interaction,
        grad_corr
    );
}

TORCH_LIBRARY_FRAGMENT(deepmd, m) {
    m.def(
        "nufft_les_frame_correction_bundle(Tensor coord, Tensor latent_charge, Tensor box, Tensor sigma, int[] nk_cpu, bool remove_self_interaction, bool need_force, bool need_virial) -> Tensor[]");
    m.impl("nufft_les_frame_correction_bundle", torch::kCUDA,
           &nufft_les_frame_correction_bundle);
    m.def(
        "nufft_les_frame_correction_backward_energy(Tensor coord, Tensor latent_charge, Tensor box, Tensor sigma, int[] nk_cpu, bool remove_self_interaction, Tensor grad_corr) -> Tensor[]");
    m.impl("nufft_les_frame_correction_backward_energy", torch::kCUDA,
           &nufft_les_frame_correction_backward_energy);
}
