#include "nerf/model.hpp"
#include "nerf/renderer.hpp"
#include "nerf/dataset.hpp"
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <filesystem>

using namespace nerf;

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
            return 1;
        }
        
        // Load configuration
        std::string config_file = argv[1];
        // TODO: Implement config loading
        
        // Set device
        torch::Device device(torch::kCUDA);
        
        // Create model
        auto model = std::make_shared<NeRFModel>(
            8,    // netdepth
            256,  // netwidth
            8,    // netdepth_fine
            256,  // netwidth_fine
            10,   // multires
            4,    // multires_views
            true  // use_viewdirs
        );
        model->to(device);
        
        // Create renderer
        auto renderer = std::make_shared<Renderer>(
            model,
            64,    // N_samples
            64,    // N_importance
            true,  // use_viewdirs
            1.0f,  // raw_noise_std
            true   // white_bkgd
        );
        
        // Create dataset
        auto dataset = std::make_shared<Dataset>(
            "./data/nerf_synthetic/lego",  // datadir
            "blender",                     // dataset_type
            8,                            // factor
            true,                         // use_viewdirs
            true                          // white_bkgd
        );
        
        // Create optimizer
        torch::optim::Adam optimizer(
            model->parameters(),
            torch::optim::AdamOptions(1e-3)
        );
        
        // Training loop
        int num_epochs = 100000;
        int batch_size = 1024;
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Get batch of rays
            auto [rays_o, rays_d, target_rgb] = dataset->get_data();
            rays_o = rays_o.to(device);
            rays_d = rays_d.to(device);
            target_rgb = target_rgb.to(device);
            
            // Forward pass
            auto [rgb_map, depth_map, acc_map, _] = renderer->render_rays(
                rays_o, rays_d, rays_d,
                dataset->get_near().to(device),
                dataset->get_far().to(device)
            );
            
            // Compute loss
            auto loss = torch::mse_loss(rgb_map, target_rgb);
            
            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            // Print progress
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
            }
            
            // Save checkpoint
            if (epoch % 1000 == 0) {
                torch::save(model, "checkpoint_" + std::to_string(epoch) + ".pt");
            }
        }
        
        // Save final model
        torch::save(model, "final_model.pt");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 