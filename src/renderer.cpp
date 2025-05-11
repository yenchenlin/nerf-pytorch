#include "nerf/renderer.hpp"
#include <cmath>

namespace nerf {

Renderer::Renderer(std::shared_ptr<NeRFModel> model,
                  int N_samples,
                  int N_importance,
                  bool use_viewdirs,
                  float raw_noise_std,
                  bool white_bkgd)
    : model_(model),
      N_samples_(N_samples),
      N_importance_(N_importance),
      use_viewdirs_(use_viewdirs),
      raw_noise_std_(raw_noise_std),
      white_bkgd_(white_bkgd) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Renderer::render_rays(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& viewdirs,
    const torch::Tensor& near,
    const torch::Tensor& far,
    bool is_fine) {
    
    // Get number of rays
    int N_rays = rays_o.size(0);
    
    // Sample points along rays
    auto t_vals = torch::linspace(0, 1, N_samples_);
    auto z_vals = near.expand({N_rays, N_samples_}) * (1 - t_vals) + 
                 far.expand({N_rays, N_samples_}) * t_vals;
    
    // Add noise to z_vals
    if (raw_noise_std_ > 0) {
        auto noise = torch::randn({N_rays, N_samples_}) * raw_noise_std_;
        z_vals = z_vals + noise;
    }
    
    // Get points along rays
    auto pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1);
    
    // Flatten points and viewdirs
    auto pts_flat = pts.reshape({-1, 3});
    auto viewdirs_flat = viewdirs.unsqueeze(1).expand({N_rays, N_samples_, 3})
                                    .reshape({-1, 3});
    
    // Get model outputs
    auto [rgb, alpha, raw] = model_->forward(pts_flat, viewdirs_flat, is_fine);
    
    // Reshape outputs
    rgb = rgb.reshape({N_rays, N_samples_, 3});
    alpha = alpha.reshape({N_rays, N_samples_});
    
    // Compute weights
    auto weights = compute_accumulated_transmittance(alpha);
    
    // Compute final RGB
    auto rgb_map = torch::sum(weights.unsqueeze(-1) * rgb, 1);
    
    // Compute depth map
    auto depth_map = torch::sum(weights * z_vals, 1);
    
    // Compute accumulated transmittance
    auto acc_map = torch::sum(weights, 1);
    
    // Add white background if needed
    if (white_bkgd_) {
        rgb_map = rgb_map + (1 - acc_map).unsqueeze(-1);
    }
    
    return std::make_tuple(rgb_map, depth_map, acc_map, weights);
}

torch::Tensor Renderer::render(
    const torch::Tensor& H,
    const torch::Tensor& W,
    const torch::Tensor& K,
    const torch::Tensor& c2w,
    const torch::Tensor& near,
    const torch::Tensor& far,
    bool is_fine) {
    
    // Create pixel coordinates
    auto i = torch::arange(H.item<int>());
    auto j = torch::arange(W.item<int>());
    auto ij = torch::meshgrid({i, j});
    auto dirs = torch::stack({
        (ij[1] - K[0][2].item<float>()) / K[0][0].item<float>(),
        -(ij[0] - K[1][2].item<float>()) / K[1][1].item<float>(),
        -torch::ones_like(ij[0])
    }, -1);
    
    // Rotate ray directions from camera frame to world frame
    auto rays_d = torch::sum(dirs.unsqueeze(-2) * c2w.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}), -1);
    
    // Normalize ray directions
    rays_d = rays_d / torch::norm(rays_d, 2, -1, true);
    
    // Get ray origins
    auto rays_o = c2w.index({torch::indexing::Slice(0, 3), 3}).expand(rays_d.sizes());
    
    // Reshape for rendering
    rays_o = rays_o.reshape({-1, 3});
    rays_d = rays_d.reshape({-1, 3});
    
    // Render rays
    auto [rgb_map, depth_map, acc_map, _] = render_rays(rays_o, rays_d, rays_d, near, far, is_fine);
    
    // Reshape outputs
    rgb_map = rgb_map.reshape({H.item<int>(), W.item<int>(), 3});
    depth_map = depth_map.reshape({H.item<int>(), W.item<int>()});
    acc_map = acc_map.reshape({H.item<int>(), W.item<int>()});
    
    return rgb_map;
}

torch::Tensor Renderer::sample_pdf(const torch::Tensor& bins, const torch::Tensor& weights, int N_samples) {
    // Normalize weights
    weights = weights + 1e-5;
    auto pdf = weights / torch::sum(weights, -1, true);
    auto cdf = torch::cumsum(pdf, -1);
    cdf = torch::cat({torch::zeros_like(cdf.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})), cdf}, -1);
    
    // Take uniform samples
    auto u = torch::rand({weights.size(0), N_samples});
    
    // Invert CDF
    auto inds = torch::searchsorted(cdf, u, true);
    auto below = torch::max(torch::zeros_like(inds), inds - 1);
    auto above = torch::min(torch::ones_like(inds) * (cdf.size(-1) - 1), inds);
    auto inds_g = torch::stack({below, above}, -1);
    
    auto cdf_g = torch::gather(cdf, -1, inds_g);
    auto bins_g = torch::gather(bins, -1, inds_g);
    
    auto denom = (cdf_g.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) -
                 cdf_g.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}));
    denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);
    auto t = (u - cdf_g.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})) / denom;
    auto samples = bins_g.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) +
                  t * (bins_g.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) -
                       bins_g.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}));
    
    return samples;
}

torch::Tensor Renderer::compute_accumulated_transmittance(const torch::Tensor& alphas) {
    auto transmittance = torch::cumprod(1 - alphas + 1e-10, -1);
    return alphas * torch::cat({torch::ones_like(transmittance.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})),
                               transmittance.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)})}, -1);
}

} // namespace nerf 