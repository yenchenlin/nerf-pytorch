#include "nerf/dataset.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace nerf {

Dataset::Dataset(const std::string& datadir,
                const std::string& dataset_type,
                int factor,
                bool use_viewdirs,
                bool white_bkgd)
    : datadir_(datadir),
      dataset_type_(dataset_type),
      factor_(factor),
      use_viewdirs_(use_viewdirs),
      white_bkgd_(white_bkgd) {
    
    if (dataset_type_ == "llff") {
        load_llff_data();
    } else if (dataset_type_ == "blender") {
        load_blender_data();
    } else {
        throw std::runtime_error("Unknown dataset type: " + dataset_type_);
    }
}

void Dataset::load_llff_data() {
    // Load images
    auto images_dir = std::filesystem::path(datadir_) / "images";
    std::vector<torch::Tensor> images;
    for (const auto& entry : std::filesystem::directory_iterator(images_dir)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            images.push_back(load_image(entry.path().string()));
        }
    }
    
    // Stack images
    auto images_tensor = torch::stack(images);
    
    // Load poses
    auto poses = load_poses((std::filesystem::path(datadir_) / "poses_bounds.npy").string());
    
    // Get image dimensions
    H_ = images_tensor.size(1);
    W_ = images_tensor.size(2);
    
    // Load camera parameters
    auto K = load_poses((std::filesystem::path(datadir_) / "hwf.npy").string());
    focal_ = K[0].item<float>();
    K_ = torch::tensor({
        {focal_, 0, W_ / 2.0f},
        {0, focal_, H_ / 2.0f},
        {0, 0, 1}
    });
    
    // Set near and far planes
    near_ = torch::tensor(0.0f);
    far_ = torch::tensor(1.0f);
}

void Dataset::load_blender_data() {
    // Load images
    auto images_dir = std::filesystem::path(datadir_) / "train";
    std::vector<torch::Tensor> images;
    for (const auto& entry : std::filesystem::directory_iterator(images_dir)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            images.push_back(load_image(entry.path().string()));
        }
    }
    
    // Stack images
    auto images_tensor = torch::stack(images);
    
    // Load poses
    auto poses = load_poses((std::filesystem::path(datadir_) / "transforms_train.json").string());
    
    // Get image dimensions
    H_ = images_tensor.size(1);
    W_ = images_tensor.size(2);
    
    // Set camera parameters
    focal_ = 138.88887889922103f;  // Default focal length for Blender dataset
    K_ = torch::tensor({
        {focal_, 0, W_ / 2.0f},
        {0, focal_, H_ / 2.0f},
        {0, 0, 1}
    });
    
    // Set near and far planes
    near_ = torch::tensor(2.0f);
    far_ = torch::tensor(6.0f);
}

torch::Tensor Dataset::load_image(const std::string& path) {
    cv::Mat img = cv::imread(path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // Resize if needed
    if (factor_ > 1) {
        cv::resize(img, img, cv::Size(), 1.0f / factor_, 1.0f / factor_);
    }
    
    // Convert to float and normalize
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0f / 255.0f);
    
    // Convert to torch tensor
    auto tensor = torch::from_blob(float_img.data, {img.rows, img.cols, 3}, torch::kFloat32);
    return tensor.clone();  // Clone to ensure memory ownership
}

torch::Tensor Dataset::load_poses(const std::string& path) {
    // This is a simplified version. In practice, you'll need to implement
    // proper loading of different pose file formats (npy, json, etc.)
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open pose file: " + path);
    }
    
    // Read file header to determine format
    std::string header;
    std::getline(file, header);
    
    if (path.find(".npy") != std::string::npos) {
        // Load numpy array
        // Implementation depends on your numpy file reading library
        throw std::runtime_error("Numpy file loading not implemented");
    } else if (path.find(".json") != std::string::npos) {
        // Load JSON
        // Implementation depends on your JSON parsing library
        throw std::runtime_error("JSON file loading not implemented");
    } else {
        throw std::runtime_error("Unsupported pose file format: " + path);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Dataset::get_data() {
    // This is a placeholder. In practice, you'll need to implement
    // proper data loading and preprocessing based on your dataset type
    return std::make_tuple(
        torch::zeros({1}),  // images
        torch::zeros({1}),  // poses
        torch::zeros({1}),  // render_poses
        torch::zeros({1}),  // hwf
        torch::zeros({1})   // i_split
    );
}

} // namespace nerf 