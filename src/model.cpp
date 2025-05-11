#include "nerf/model.hpp"
#include <cmath>

namespace nerf {

NeRFModel::NeRFModel(int netdepth, int netwidth, int netdepth_fine, int netwidth_fine,
                     int multires, int multires_views, bool use_viewdirs)
    : netdepth_(netdepth), netwidth_(netwidth),
      netdepth_fine_(netdepth_fine), netwidth_fine_(netwidth_fine),
      multires_(multires), multires_views_(multires_views),
      use_viewdirs_(use_viewdirs) {
    
    // Create the main network
    std::vector<torch::nn::Linear> layers;
    int input_ch = 3 + 2 * multires_ * 3;  // 3D position + positional encoding
    
    for (int i = 0; i < netdepth_; i++) {
        int in_channels = (i == 0) ? input_ch : netwidth_;
        layers.push_back(torch::nn::Linear(in_channels, netwidth_));
        register_module("lin" + std::to_string(i), layers.back());
    }
    
    net_ = torch::nn::Sequential(layers);
    register_module("net", net_);
    
    // Create the fine network
    std::vector<torch::nn::Linear> fine_layers;
    for (int i = 0; i < netdepth_fine_; i++) {
        int in_channels = (i == 0) ? input_ch : netwidth_fine_;
        fine_layers.push_back(torch::nn::Linear(in_channels, netwidth_fine_));
        register_module("lin_fine" + std::to_string(i), fine_layers.back());
    }
    
    net_fine_ = torch::nn::Sequential(fine_layers);
    register_module("net_fine", net_fine_);
    
    // Create view-dependent networks if needed
    if (use_viewdirs_) {
        int input_ch_views = 3 + 2 * multires_views_ * 3;  // View direction + positional encoding
        viewdirs_net_ = torch::nn::Linear(netwidth_ + input_ch_views, netwidth_ / 2);
        viewdirs_net_fine_ = torch::nn::Linear(netwidth_fine_ + input_ch_views, netwidth_fine_ / 2);
        register_module("viewdirs_net", viewdirs_net_);
        register_module("viewdirs_net_fine", viewdirs_net_fine_);
    }
}

torch::Tensor NeRFModel::embed_fn(const torch::Tensor& inputs) {
    auto x = inputs;
    std::vector<torch::Tensor> embeds;
    embeds.push_back(x);
    
    for (int i = 0; i < multires_; i++) {
        for (int j = 0; j < 3; j++) {
            float freq = std::pow(2.0f, i);
            float phase = (j % 2) * M_PI / 2;
            embeds.push_back(torch::sin(freq * x.index({torch::indexing::Slice(), j}) + phase));
            embeds.push_back(torch::cos(freq * x.index({torch::indexing::Slice(), j}) + phase));
        }
    }
    
    return torch::cat(embeds, -1);
}

torch::Tensor NeRFModel::embeddirs_fn(const torch::Tensor& inputs) {
    auto x = inputs;
    std::vector<torch::Tensor> embeds;
    embeds.push_back(x);
    
    for (int i = 0; i < multires_views_; i++) {
        for (int j = 0; j < 3; j++) {
            float freq = std::pow(2.0f, i);
            float phase = (j % 2) * M_PI / 2;
            embeds.push_back(torch::sin(freq * x.index({torch::indexing::Slice(), j}) + phase));
            embeds.push_back(torch::cos(freq * x.index({torch::indexing::Slice(), j}) + phase));
        }
    }
    
    return torch::cat(embeds, -1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NeRFModel::forward(
    const torch::Tensor& inputs_flat,
    const torch::Tensor& viewdirs,
    bool is_fine) {
    
    auto inputs_embedded = embed_fn(inputs_flat);
    auto net = is_fine ? net_fine_ : net_;
    auto x = net->forward(inputs_embedded);
    
    if (use_viewdirs_ && viewdirs.numel() > 0) {
        auto viewdirs_embedded = embeddirs_fn(viewdirs);
        auto viewdirs_net = is_fine ? viewdirs_net_fine_ : viewdirs_net_;
        x = torch::cat({x, viewdirs_embedded}, -1);
        x = viewdirs_net->forward(x);
    }
    
    auto rgb = torch::sigmoid(x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));
    auto alpha = torch::sigmoid(x.index({torch::indexing::Slice(), 3}));
    auto raw = x;
    
    return std::make_tuple(rgb, alpha, raw);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NeRFModel::get_outputs(
    const torch::Tensor& inputs_flat,
    const torch::Tensor& viewdirs,
    bool is_fine) {
    return forward(inputs_flat, viewdirs, is_fine);
}

} // namespace nerf 