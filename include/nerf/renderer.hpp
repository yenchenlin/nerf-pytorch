#pragma once

#include "model.hpp"
#include <torch/torch.h>
#include <memory>
#include <tuple>

namespace nerf {

class Renderer {
public:
    Renderer(std::shared_ptr<NeRFModel> model,
             int N_samples = 64,
             int N_importance = 64,
             bool use_viewdirs = true,
             float raw_noise_std = 0.0f,
             bool white_bkgd = false);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_rays(
        const torch::Tensor& rays_o,
        const torch::Tensor& rays_d,
        const torch::Tensor& viewdirs,
        const torch::Tensor& near,
        const torch::Tensor& far,
        bool is_fine = false);

    torch::Tensor render(
        const torch::Tensor& H,
        const torch::Tensor& W,
        const torch::Tensor& K,
        const torch::Tensor& c2w,
        const torch::Tensor& near,
        const torch::Tensor& far,
        bool is_fine = false);

private:
    std::shared_ptr<NeRFModel> model_;
    int N_samples_;
    int N_importance_;
    bool use_viewdirs_;
    float raw_noise_std_;
    bool white_bkgd_;

    torch::Tensor sample_pdf(const torch::Tensor& bins, const torch::Tensor& weights, int N_samples);
    torch::Tensor compute_accumulated_transmittance(const torch::Tensor& alphas);
};

} // namespace nerf