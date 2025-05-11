#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace nerf {

class Dataset {
public:
    Dataset(const std::string& datadir,
            const std::string& dataset_type,
            int factor = 8,
            bool use_viewdirs = true,
            bool white_bkgd = false);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    get_data();

    int get_H() const { return H_; }
    int get_W() const { return W_; }
    int get_K() const { return K_; }
    int get_focal() const { return focal_; }
    torch::Tensor get_near() const { return near_; }
    torch::Tensor get_far() const { return far_; }

private:
    std::string datadir_;
    std::string dataset_type_;
    int factor_;
    bool use_viewdirs_;
    bool white_bkgd_;

    int H_;
    int W_;
    int K_;
    float focal_;
    torch::Tensor near_;
    torch::Tensor far_;

    void load_llff_data();
    void load_blender_data();
    torch::Tensor load_image(const std::string& path);
    torch::Tensor load_poses(const std::string& path);
};

} // namespace nerf