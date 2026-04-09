#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

template <typename scalar_t>
struct ComplexType;

template <>
struct ComplexType<float> {
    using type = float2;
    __device__ static inline type make(float x, float y) {
        return make_float2(x, y);
    }
};

template <>
struct ComplexType<double> {
    using type = double2;
    __device__ static inline type make(double x, double y) {
        return make_double2(x, y);
    }
};

template <typename scalar_t>
__device__ inline typename ComplexType<scalar_t>::type make_complex(scalar_t x, scalar_t y) {
    return ComplexType<scalar_t>::make(x, y);
}

__device__ inline int fft_mode_index(int i, int nk, int dim) {
    return (i <= nk) ? i : (i - dim);
}

// CUDA kernel to perform frequency space modulation for LES correction:
// E_num = Sum_k [ |rho(k)|^2 * exp(-0.5 * sigma^2 * |k|^2) / |k|^2 ]
template <typename scalar_t>
__global__ void les_modulate_kernel_t(
    int n_k_x,
    int n_k_y,
    int n_k_z,
    int dim_x,
    int dim_y,
    int dim_z,
    const typename ComplexType<scalar_t>::type* grid_k,
    scalar_t sigma_sq,
    const scalar_t* cell_inv,
    scalar_t* out_corr,
    scalar_t* out_kfac_sum) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx_i = fft_mode_index(ix, n_k_x, dim_x);
    int ny_i = fft_mode_index(iy, n_k_y, dim_y);
    int nz_i = fft_mode_index(iz, n_k_z, dim_z);
    scalar_t nx = static_cast<scalar_t>(nx_i);
    scalar_t ny = static_cast<scalar_t>(ny_i);
    scalar_t nz = static_cast<scalar_t>(nz_i);

    const scalar_t two_pi = static_cast<scalar_t>(2.0 * 3.14159265358979323846);

    size_t idx = ix + dim_x * (iy + dim_y * iz);
    auto rho = grid_k[idx];
    scalar_t rho_sq = rho.x * rho.x + rho.y * rho.y;

    scalar_t kcart_x =
        two_pi * (cell_inv[0] * nx + cell_inv[1] * ny + cell_inv[2] * nz);
    scalar_t kcart_y =
        two_pi * (cell_inv[3] * nx + cell_inv[4] * ny + cell_inv[5] * nz);
    scalar_t kcart_z =
        two_pi * (cell_inv[6] * nx + cell_inv[7] * ny + cell_inv[8] * nz);

    scalar_t k_sq = kcart_x * kcart_x + kcart_y * kcart_y + kcart_z * kcart_z;
    if (k_sq == static_cast<scalar_t>(0)) return;

    scalar_t kfac = exp(static_cast<scalar_t>(-0.5) * sigma_sq * k_sq) / k_sq;

    atomicAdd(&out_corr[0], rho_sq * kfac);
    atomicAdd(&out_kfac_sum[0], kfac);
}

// Build grad-convolution grids for type-2 NUFFT force evaluation.
// grad_conv = i * k_cart * (kfac * rho)
template <typename scalar_t>
__global__ void les_build_gradconv_kernel_t(
    int n_k_x,
    int n_k_y,
    int n_k_z,
    int dim_x,
    int dim_y,
    int dim_z,
    const typename ComplexType<scalar_t>::type* grid_k,
    scalar_t sigma_sq,
    const scalar_t* cell_inv,
    typename ComplexType<scalar_t>::type* out_gradconv_x,
    typename ComplexType<scalar_t>::type* out_gradconv_y,
    typename ComplexType<scalar_t>::type* out_gradconv_z
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx_i = fft_mode_index(ix, n_k_x, dim_x);
    int ny_i = fft_mode_index(iy, n_k_y, dim_y);
    int nz_i = fft_mode_index(iz, n_k_z, dim_z);
    scalar_t nx = static_cast<scalar_t>(nx_i);
    scalar_t ny = static_cast<scalar_t>(ny_i);
    scalar_t nz = static_cast<scalar_t>(nz_i);

    const scalar_t two_pi = static_cast<scalar_t>(2.0 * 3.14159265358979323846);
    size_t idx = ix + dim_x * (iy + dim_y * iz);

    auto rho = grid_k[idx];

    scalar_t kcart_x = two_pi * (cell_inv[0] * nx + cell_inv[1] * ny + cell_inv[2] * nz);
    scalar_t kcart_y = two_pi * (cell_inv[3] * nx + cell_inv[4] * ny + cell_inv[5] * nz);
    scalar_t kcart_z = two_pi * (cell_inv[6] * nx + cell_inv[7] * ny + cell_inv[8] * nz);

    scalar_t k_sq = kcart_x * kcart_x + kcart_y * kcart_y + kcart_z * kcart_z;
    if (k_sq == static_cast<scalar_t>(0)) {
        out_gradconv_x[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        out_gradconv_y[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        out_gradconv_z[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        return;
    }

    scalar_t kfac = exp(static_cast<scalar_t>(-0.5) * sigma_sq * k_sq) / k_sq;

    scalar_t conv_r = kfac * rho.x;
    scalar_t conv_i = kfac * rho.y;

    out_gradconv_x[idx] = make_complex<scalar_t>(-kcart_x * conv_i, kcart_x * conv_r);
    out_gradconv_y[idx] = make_complex<scalar_t>(-kcart_y * conv_i, kcart_y * conv_r);
    out_gradconv_z[idx] = make_complex<scalar_t>(-kcart_z * conv_i, kcart_z * conv_r);
}

void launch_les_modulate_kernel(
    int nkx,
    int nky,
    int nkz,
    const double* coeff_ptr,
    double sigma_sq,
    const double* cell_inv_ptr,
    double* out_corr_ptr,
    double* out_kfac_sum_ptr,
    cudaStream_t stream) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    les_modulate_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const double2*)coeff_ptr,
        sigma_sq,
        cell_inv_ptr,
        out_corr_ptr,
        out_kfac_sum_ptr);
}

void launch_les_modulate_kernel(
    int nkx,
    int nky,
    int nkz,
    const float* coeff_ptr,
    float sigma_sq,
    const float* cell_inv_ptr,
    float* out_corr_ptr,
    float* out_kfac_sum_ptr,
    cudaStream_t stream) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    les_modulate_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const float2*)coeff_ptr,
        sigma_sq,
        cell_inv_ptr,
        out_corr_ptr,
        out_kfac_sum_ptr);
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    les_build_gradconv_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const double2*)coeff_ptr,
        sigma_sq,
        cell_inv_ptr,
        (double2*)gradconv_x_ptr,
        (double2*)gradconv_y_ptr,
        (double2*)gradconv_z_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    les_build_gradconv_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const float2*)coeff_ptr,
        sigma_sq,
        cell_inv_ptr,
        (float2*)gradconv_x_ptr,
        (float2*)gradconv_y_ptr,
        (float2*)gradconv_z_ptr
    );
}

// Backward helper for energy-only autograd:
// 1) Build conv_grid = kfac * rho for latent-charge gradient via type-2 NUFFT.
// 2) Accumulate scalar reductions for sigma gradient and self-term correction.
template <typename scalar_t>
__global__ void les_backward_energy_kernel_t(
    int n_k_x,
    int n_k_y,
    int n_k_z,
    int dim_x,
    int dim_y,
    int dim_z,
    const typename ComplexType<scalar_t>::type* grid_k,
    scalar_t sigma,
    scalar_t sigma_sq,
    const scalar_t* cell_inv,
    typename ComplexType<scalar_t>::type* conv_grid,
    scalar_t* out_sum_kfac,
    scalar_t* out_sum_dsigma_rho,
    scalar_t* out_sum_dsigma
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx_i = fft_mode_index(ix, n_k_x, dim_x);
    int ny_i = fft_mode_index(iy, n_k_y, dim_y);
    int nz_i = fft_mode_index(iz, n_k_z, dim_z);
    scalar_t nx = static_cast<scalar_t>(nx_i);
    scalar_t ny = static_cast<scalar_t>(ny_i);
    scalar_t nz = static_cast<scalar_t>(nz_i);

    const scalar_t two_pi = static_cast<scalar_t>(2.0 * 3.14159265358979323846);
    size_t idx = ix + dim_x * (iy + dim_y * iz);

    auto rho = grid_k[idx];
    scalar_t rho_sq = rho.x * rho.x + rho.y * rho.y;

    scalar_t kcart_x = two_pi * (cell_inv[0] * nx + cell_inv[1] * ny + cell_inv[2] * nz);
    scalar_t kcart_y = two_pi * (cell_inv[3] * nx + cell_inv[4] * ny + cell_inv[5] * nz);
    scalar_t kcart_z = two_pi * (cell_inv[6] * nx + cell_inv[7] * ny + cell_inv[8] * nz);

    scalar_t k_sq = kcart_x * kcart_x + kcart_y * kcart_y + kcart_z * kcart_z;
    if (k_sq == static_cast<scalar_t>(0)) {
        conv_grid[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        return;
    }

    scalar_t exp_term = exp(static_cast<scalar_t>(-0.5) * sigma_sq * k_sq);
    scalar_t kfac = exp_term / k_sq;
    scalar_t d_sigma = -sigma * exp_term;

    conv_grid[idx] = make_complex<scalar_t>(kfac * rho.x, kfac * rho.y);

    atomicAdd(&out_sum_kfac[0], kfac);
    atomicAdd(&out_sum_dsigma_rho[0], rho_sq * d_sigma);
    atomicAdd(&out_sum_dsigma[0], d_sigma);
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    les_backward_energy_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const double2*)coeff_ptr,
        sigma,
        sigma_sq,
        cell_inv_ptr,
        (double2*)conv_grid_ptr,
        sum_kfac_ptr,
        sum_dsigma_rho_ptr,
        sum_dsigma_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    les_backward_energy_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const float2*)coeff_ptr,
        sigma,
        sigma_sq,
        cell_inv_ptr,
        (float2*)conv_grid_ptr,
        sum_kfac_ptr,
        sum_dsigma_rho_ptr,
        sum_dsigma_ptr
    );
}
