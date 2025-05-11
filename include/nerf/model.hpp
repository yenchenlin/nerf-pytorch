#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

namespace nerf {

class NeRFModel : public torch::nn::Module {
public:
    NeRFModel(int netdepth = 8, int netwidth = 256, int netdepth_fine = 8, int netwidth_fine = 256,
              int multires = 10, int multires_views = 4, bool use_viewdirs = true);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& inputs_flat,
        const torch::Tensor& viewdirs = torch::Tensor(),
        bool is_fine = false);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_outputs(
        const torch::Tensor& inputs_flat,
        const torch::Tensor& viewdirs = torch::Tensor(),
        bool is_fine = false);

private:
    int netdepth_;
    int netwidth_;
    int netdepth_fine_;
    int netwidth_fine_;
    int multires_;
    int multires_views_;
    bool use_viewdirs_;

    torch::nn::Sequential net_{nullptr};
    torch::nn::Sequential net_fine_{nullptr};
    torch::nn::Linear viewdirs_net_{nullptr};
    torch::nn::Linear viewdirs_net_fine_{nullptr};

    torch::Tensor embed_fn(const torch::Tensor& inputs);
    torch::Tensor embeddirs_fn(const torch::Tensor& inputs);
};

} // namespace nerf