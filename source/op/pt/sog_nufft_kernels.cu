#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>
#include <type_traits>

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

template <typename scalar_t>
__global__ void sog_modulate_kernel_t(
    int n_k_x, int n_k_y, int n_k_z, 
    int dim_x, int dim_y, int dim_z,
    const typename ComplexType<scalar_t>::type* grid_k,
    const scalar_t* amp,
    const scalar_t* bw_sq,
    int num_bandwidth,
    const scalar_t* cell_inv,
    scalar_t* out_corr,
    scalar_t* out_kfac_sum
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx = (ix <= n_k_x) ? ix : ix - (2 * n_k_x + 1);
    int ny = (iy <= n_k_y) ? iy : iy - (2 * n_k_y + 1);
    int nz = (iz <= n_k_z) ? iz : iz - (2 * n_k_z + 1);

    const scalar_t two_pi = static_cast<scalar_t>(2.0 * 3.14159265358979323846);

    size_t idx = ix + dim_x * (iy + dim_y * iz);

    auto rho = grid_k[idx];
    scalar_t rho_sq = rho.x * rho.x + rho.y * rho.y;

    scalar_t kcart_x = two_pi * (cell_inv[0] * nx + cell_inv[1] * ny + cell_inv[2] * nz);
    scalar_t kcart_y = two_pi * (cell_inv[3] * nx + cell_inv[4] * ny + cell_inv[5] * nz);
    scalar_t kcart_z = two_pi * (cell_inv[6] * nx + cell_inv[7] * ny + cell_inv[8] * nz);

    scalar_t k_sq = kcart_x * kcart_x + kcart_y * kcart_y + kcart_z * kcart_z;

    if (k_sq == static_cast<scalar_t>(0)) return;

    scalar_t kfac = static_cast<scalar_t>(0);
    for (int j = 0; j < num_bandwidth; j++) {
        scalar_t b2 = bw_sq[j];
        kfac += b2 * exp(static_cast<scalar_t>(-0.5) * b2 * k_sq);
    }
    kfac *= amp[0];

    scalar_t e_val = rho_sq * kfac;

    atomicAdd(&out_corr[0], e_val);
    atomicAdd(&out_kfac_sum[0], kfac);
}

template <typename scalar_t>
__global__ void sog_build_gradconv_kernel_t(
    int n_k_x,
    int n_k_y,
    int n_k_z,
    int dim_x,
    int dim_y,
    int dim_z,
    const typename ComplexType<scalar_t>::type* grid_k,
    const scalar_t* amp,
    const scalar_t* bw_sq,
    int num_bandwidth,
    const scalar_t* cell_inv,
    typename ComplexType<scalar_t>::type* out_gradconv_x,
    typename ComplexType<scalar_t>::type* out_gradconv_y,
    typename ComplexType<scalar_t>::type* out_gradconv_z
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx = (ix <= n_k_x) ? ix : ix - (2 * n_k_x + 1);
    int ny = (iy <= n_k_y) ? iy : iy - (2 * n_k_y + 1);
    int nz = (iz <= n_k_z) ? iz : iz - (2 * n_k_z + 1);

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

    scalar_t kfac = static_cast<scalar_t>(0);
    for (int j = 0; j < num_bandwidth; j++) {
        scalar_t b2 = bw_sq[j];
        kfac += b2 * exp(static_cast<scalar_t>(-0.5) * b2 * k_sq);
    }
    kfac *= amp[0];

    scalar_t conv_r = kfac * rho.x;
    scalar_t conv_i = kfac * rho.y;

    out_gradconv_x[idx] = make_complex<scalar_t>(-kcart_x * conv_i, kcart_x * conv_r);
    out_gradconv_y[idx] = make_complex<scalar_t>(-kcart_y * conv_i, kcart_y * conv_r);
    out_gradconv_z[idx] = make_complex<scalar_t>(-kcart_z * conv_i, kcart_z * conv_r);
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_modulate_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx, nky, nkz, dim_x, dim_y, dim_z,
        (const double2*)coeff_ptr,
        amp_ptr, bw_sq_ptr, num_bandwidth, cell_inv_ptr, out_corr_ptr, out_kfac_sum_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_modulate_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx, nky, nkz, dim_x, dim_y, dim_z,
        (const float2*)coeff_ptr,
        amp_ptr, bw_sq_ptr, num_bandwidth, cell_inv_ptr, out_corr_ptr, out_kfac_sum_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_build_gradconv_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const double2*)coeff_ptr,
        amp_ptr,
        bw_sq_ptr,
        num_bandwidth,
        cell_inv_ptr,
        (double2*)gradconv_x_ptr,
        (double2*)gradconv_y_ptr,
        (double2*)gradconv_z_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_build_gradconv_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const float2*)coeff_ptr,
        amp_ptr,
        bw_sq_ptr,
        num_bandwidth,
        cell_inv_ptr,
        (float2*)gradconv_x_ptr,
        (float2*)gradconv_y_ptr,
        (float2*)gradconv_z_ptr
    );
}

template <typename scalar_t>
__global__ void sog_backward_energy_kernel_t(
    int n_k_x,
    int n_k_y,
    int n_k_z,
    int dim_x,
    int dim_y,
    int dim_z,
    const typename ComplexType<scalar_t>::type* grid_k,
    const scalar_t* amp,
    const scalar_t* bw_sq,
    int num_bandwidth,
    const scalar_t* cell_inv,
    typename ComplexType<scalar_t>::type* conv_grid,
    scalar_t* out_sum_base_rho,
    scalar_t* out_sum_base,
    scalar_t* out_sum_dbw_rho,
    scalar_t* out_sum_dbw
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx = (ix <= n_k_x) ? ix : ix - (2 * n_k_x + 1);
    int ny = (iy <= n_k_y) ? iy : iy - (2 * n_k_y + 1);
    int nz = (iz <= n_k_z) ? iz : iz - (2 * n_k_z + 1);

    const scalar_t two_pi = static_cast<scalar_t>(2.0 * 3.14159265358979323846);
    size_t idx = ix + dim_x * (iy + dim_y * iz);

    auto rho = grid_k[idx];

    scalar_t kcart_x = two_pi * (cell_inv[0] * nx + cell_inv[1] * ny + cell_inv[2] * nz);
    scalar_t kcart_y = two_pi * (cell_inv[3] * nx + cell_inv[4] * ny + cell_inv[5] * nz);
    scalar_t kcart_z = two_pi * (cell_inv[6] * nx + cell_inv[7] * ny + cell_inv[8] * nz);

    scalar_t k_sq = kcart_x * kcart_x + kcart_y * kcart_y + kcart_z * kcart_z;
    if (k_sq == static_cast<scalar_t>(0)) {
        conv_grid[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        return;
    }

    scalar_t rho_sq = rho.x * rho.x + rho.y * rho.y;

    scalar_t base_sum = static_cast<scalar_t>(0);
    for (int j = 0; j < num_bandwidth; j++) {
        scalar_t b2 = bw_sq[j];
        base_sum += b2 * exp(static_cast<scalar_t>(-0.5) * b2 * k_sq);
    }

    scalar_t kfac = amp[0] * base_sum;
    conv_grid[idx] = make_complex<scalar_t>(kfac * rho.x, kfac * rho.y);

    atomicAdd(&out_sum_base_rho[0], rho_sq * base_sum);
    atomicAdd(&out_sum_base[0], base_sum);

    for (int j = 0; j < num_bandwidth; j++) {
        scalar_t bw = sqrt(bw_sq[j]);
        scalar_t dterm = amp[0] * bw * exp(static_cast<scalar_t>(-0.5) * bw_sq[j] * k_sq) *
                         (static_cast<scalar_t>(2.0) - bw_sq[j] * k_sq);
        atomicAdd(&out_sum_dbw_rho[j], rho_sq * dterm);
        atomicAdd(&out_sum_dbw[j], dterm);
    }
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_backward_energy_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const double2*)coeff_ptr,
        amp_ptr,
        bw_sq_ptr,
        num_bandwidth,
        cell_inv_ptr,
        (double2*)conv_grid_ptr,
        sum_base_rho_ptr,
        sum_base_ptr,
        sum_dbw_rho_ptr,
        sum_dbw_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_backward_energy_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const float2*)coeff_ptr,
        amp_ptr,
        bw_sq_ptr,
        num_bandwidth,
        cell_inv_ptr,
        (float2*)conv_grid_ptr,
        sum_base_rho_ptr,
        sum_base_ptr,
        sum_dbw_rho_ptr,
        sum_dbw_ptr
    );
}

template <typename scalar_t>
__global__ void sog_backward_force_prepare_kernel_t(
    int n_k_x,
    int n_k_y,
    int n_k_z,
    int dim_x,
    int dim_y,
    int dim_z,
    const typename ComplexType<scalar_t>::type* rho_grid,
    const typename ComplexType<scalar_t>::type* sminus_x_grid,
    const typename ComplexType<scalar_t>::type* sminus_y_grid,
    const typename ComplexType<scalar_t>::type* sminus_z_grid,
    const scalar_t* amp,
    const scalar_t* bw_sq,
    int num_bandwidth,
    const scalar_t* cell_inv,
    scalar_t inv_volume,
    scalar_t* out_sum_base_force,
    scalar_t* out_sum_dbw_force,
    typename ComplexType<scalar_t>::type* out_indirect_x,
    typename ComplexType<scalar_t>::type* out_indirect_y,
    typename ComplexType<scalar_t>::type* out_indirect_z
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;

    int nx = (ix <= n_k_x) ? ix : ix - (2 * n_k_x + 1);
    int ny = (iy <= n_k_y) ? iy : iy - (2 * n_k_y + 1);
    int nz = (iz <= n_k_z) ? iz : iz - (2 * n_k_z + 1);

    const scalar_t two_pi = static_cast<scalar_t>(2.0 * 3.14159265358979323846);
    size_t idx = ix + dim_x * (iy + dim_y * iz);

    scalar_t kcart_x = two_pi * (cell_inv[0] * nx + cell_inv[1] * ny + cell_inv[2] * nz);
    scalar_t kcart_y = two_pi * (cell_inv[3] * nx + cell_inv[4] * ny + cell_inv[5] * nz);
    scalar_t kcart_z = two_pi * (cell_inv[6] * nx + cell_inv[7] * ny + cell_inv[8] * nz);

    scalar_t k_sq = kcart_x * kcart_x + kcart_y * kcart_y + kcart_z * kcart_z;
    if (k_sq == static_cast<scalar_t>(0)) {
        out_indirect_x[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        out_indirect_y[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        out_indirect_z[idx] = make_complex<scalar_t>(static_cast<scalar_t>(0), static_cast<scalar_t>(0));
        return;
    }

    scalar_t base_sum = static_cast<scalar_t>(0);
    for (int j = 0; j < num_bandwidth; j++) {
        scalar_t b2 = bw_sq[j];
        base_sum += b2 * exp(static_cast<scalar_t>(-0.5) * b2 * k_sq);
    }
    scalar_t kfac = amp[0] * base_sum;

    auto sxm = sminus_x_grid[idx];
    auto sym = sminus_y_grid[idx];
    auto szm = sminus_z_grid[idx];
    auto sxp = make_complex<scalar_t>(sxm.x, -sxm.y);
    auto syp = make_complex<scalar_t>(sym.x, -sym.y);
    auto szp = make_complex<scalar_t>(szm.x, -szm.y);

    scalar_t z_re = kcart_x * sxp.x + kcart_y * syp.x + kcart_z * szp.x;
    scalar_t z_im = kcart_x * sxp.y + kcart_y * syp.y + kcart_z * szp.y;
    scalar_t tmp_re = -z_im;
    scalar_t tmp_im = z_re;

    auto rho = rho_grid[idx];
    scalar_t rho_tmp_re = rho.x * tmp_re - rho.y * tmp_im;
    scalar_t dldkfac = -inv_volume * rho_tmp_re;

    atomicAdd(&out_sum_base_force[0], dldkfac * base_sum);
    for (int j = 0; j < num_bandwidth; j++) {
        scalar_t b2 = bw_sq[j];
        scalar_t bw = sqrt(b2);
        scalar_t dterm = amp[0] * bw * exp(static_cast<scalar_t>(-0.5) * b2 * k_sq) *
                         (static_cast<scalar_t>(2.0) - b2 * k_sq);
        atomicAdd(&out_sum_dbw_force[j], dldkfac * dterm);
    }

    scalar_t scale_x = kcart_x * kfac;
    scalar_t scale_y = kcart_y * kfac;
    scalar_t scale_z = kcart_z * kfac;
    out_indirect_x[idx] = make_complex<scalar_t>(scale_x * sxm.y, -scale_x * sxm.x);
    out_indirect_y[idx] = make_complex<scalar_t>(scale_y * sym.y, -scale_y * sym.x);
    out_indirect_z[idx] = make_complex<scalar_t>(scale_z * szm.y, -scale_z * szm.x);
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_backward_force_prepare_kernel_t<double><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const double2*)rho_grid_ptr,
        (const double2*)sminus_x_grid_ptr,
        (const double2*)sminus_y_grid_ptr,
        (const double2*)sminus_z_grid_ptr,
        amp_ptr,
        bw_sq_ptr,
        num_bandwidth,
        cell_inv_ptr,
        inv_volume,
        sum_base_force_ptr,
        sum_dbw_force_ptr,
        (double2*)indirect_x_ptr,
        (double2*)indirect_y_ptr,
        (double2*)indirect_z_ptr
    );
}

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
) {
    int dim_x = 2 * nkx + 1;
    int dim_y = 2 * nky + 1;
    int dim_z = 2 * nkz + 1;

    dim3 block(8, 8, 8);
    dim3 grid((dim_x + block.x - 1) / block.x,
              (dim_y + block.y - 1) / block.y,
              (dim_z + block.z - 1) / block.z);

    sog_backward_force_prepare_kernel_t<float><<<grid, block, 0, stream>>>(
        nkx,
        nky,
        nkz,
        dim_x,
        dim_y,
        dim_z,
        (const float2*)rho_grid_ptr,
        (const float2*)sminus_x_grid_ptr,
        (const float2*)sminus_y_grid_ptr,
        (const float2*)sminus_z_grid_ptr,
        amp_ptr,
        bw_sq_ptr,
        num_bandwidth,
        cell_inv_ptr,
        inv_volume,
        sum_base_force_ptr,
        sum_dbw_force_ptr,
        (float2*)indirect_x_ptr,
        (float2*)indirect_y_ptr,
        (float2*)indirect_z_ptr
    );
}

