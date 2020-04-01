import numpy as np
import tensorflow as tf
import torch
tf.compat.v1.enable_eager_execution()


### Model 
def check_model():
    from run_nerf_helpers import init_nerf_model

    model = init_nerf_model(use_viewdirs=True)
    print(model.summary())

    print("--- Pytorch ---")

    from run_nerf_helpers_torch import NeRF
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_torch = NeRF(use_viewdirs=True)
    print(model_torch)
    print(f"Number of parameters: {count_parameters(model_torch)}")


def check_get_rays():
    from load_blender import load_blender_data as load_blender_data_tf
    from run_nerf_helpers import get_rays

    datadir = './test_data/nerf_synthetic/lego'
    half_res = True
    testskip = 1
    white_bkgd = True

    images, poses, render_poses, hwf, i_split = load_blender_data_tf(datadir, half_res, testskip)
    if white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

    i_train, i_val, i_test = i_split
    near = 2.
    far = 6.

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    img_i = np.random.choice(i_train)
    target = images[img_i]
    pose = poses[img_i, :3,:4]

    N_rand = 2
    print(f"H, W, focal: {H}, {W}, {focal}")
    rays_o, rays_d = get_rays(H, W, focal, pose)
    print(f"rays_d, rays_o: {rays_d.shape}, {rays_o.shape}")

    coords = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), -1)
    coords = tf.reshape(coords, [-1,2])    
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
    print(f"select_inds before gather: {select_inds}")
    select_inds = tf.gather_nd(coords, select_inds[:,tf.newaxis])
    print(f"select_inds after gather: {select_inds}")
    
    rays_o = tf.gather_nd(rays_o, select_inds)
    rays_d = tf.gather_nd(rays_d, select_inds)
    batch_rays = tf.stack([rays_o, rays_d], 0)
    target_s = tf.gather_nd(target, select_inds)

    
def check_preprocessing_one_image():
    from run_nerf_helpers_torch import get_rays, get_rays_np
    
    H, W, focal = int(400 / 40), int(400 / 40), 555.555 / 40
    hwf = [H, W, focal]
    pose = np.array([
        [-0.9305, 0.1170, -0.3469, -1.3986],
        [-0.3661, -0.2975, 0.8817, 3.554],
        [0, 0.9475, 0.3197, 1.288]
    ])

    # Sample inds of pixels
    N_rand = 10
    select_inds = np.random.choice(H, size=[N_rand], replace=False)
    
    # tf
    rays_o, rays_d = get_rays_np(H, W, focal, pose)
    coords = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), -1)
    coords = tf.reshape(coords, [-1,2])
    select_coords = tf.gather_nd(coords, select_inds[:,tf.newaxis])
    rays_o = tf.gather_nd(rays_o, select_coords)
    rays_d = tf.gather_nd(rays_d, select_coords)
    rays_d = tf.cast(rays_d, tf.float64)
    batch_rays = tf.stack([rays_o, rays_d], 0)

    # torch
    rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    batch_rays_torch = torch.stack([rays_o, rays_d], 0)

    assert np.allclose(batch_rays, batch_rays_torch.numpy())

check_preprocessing_one_image()