import pytest
import numpy as np
import torch
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def test_positional_encoding():
    from run_nerf_helpers_torch import get_embedder as get_embedder_torch
    from run_nerf_helpers import get_embedder as get_embedder_tf
    
    multires = 10
    i_embed = 0
    x = np.arange(2).astype(np.float32)

    # Pytorch
    embed_fn, input_ch = get_embedder_torch(multires, i_embed)
    with torch.no_grad():
        x_torch = torch.tensor(x)
        y_torch = embed_fn(x_torch).numpy()
    
    # Tensorflow
    embed_fn, input_ch = get_embedder_tf(multires, i_embed)
    x_tf = tf.convert_to_tensor(x)
    y_tf = embed_fn(x_tf).numpy()
    
    assert np.allclose(y_torch, y_tf), "Positional encoding is wrong!"


def test_get_rays():
    from run_nerf_helpers_torch import get_rays, get_rays_np
    H, W, focal = int(378/18), int(504/18), 407.5658/18
    hwf = [H, W, focal]
    pose = np.array([
        [-0.9305, 0.1170, -0.3469, -1.3986],
        [-0.3661, -0.2975, 0.8817, 3.554],
        [0, 0.9475, 0.3197, 1.288]
    ])

    rays_o_np, rays_d_np = get_rays_np(H, W, focal, pose)
    rays_o_torch, rays_d_torch = get_rays(H, W, focal, torch.Tensor(pose))

    assert np.allclose(rays_o_np, rays_o_torch.numpy())
    assert np.allclose(rays_d_np, rays_d_torch.numpy())


def test_sample_pdf():
    from run_nerf_helpers_torch import sample_pdf as sample_pdf_torch
    from run_nerf_helpers import sample_pdf as sample_pdf_tf
    
    N_samples = 5
    bins = np.array([
        [0., 1., 2., 4.],
        [2., 4., 6., 8.]
    ])
    weights = np.array([
        [1.0, 1.0, 1.0],
        [0.5, 1.0, 0.5]
    ])

    bins_tf = tf.cast(bins, tf.float32)
    weights_tf = tf.cast(weights, tf.float32)
    bins_torch = torch.Tensor(bins)
    weights_torch = torch.Tensor(weights)
    
    samples_tf = sample_pdf_tf(bins_tf, weights_tf, N_samples, pytest=True)
    samples_torch = sample_pdf_torch(bins_torch, weights_torch, N_samples, pytest=True)

    assert np.allclose(samples_tf.numpy(), samples_torch.numpy())


def test_model_forward_backward():
    # Only run this test if CUDA is available
    assert torch.cuda.is_available()
    
    # Hyperparams
    use_viewdirs = True
    multires = 10
    multires_views = 4
    i_embed = 0
    N_importance = 64

    # Prepare data
    inputs = np.random.rand(10, 5, 3)  # (batch_size, N_importance, xyz)
    viewdirs = np.random.rand(10, 3)   # (batch_size, view_dir)

    inputs_tf = tf.cast(inputs, tf.float32)
    viewdirs_tf = tf.cast(viewdirs, tf.float32)

    inputs_torch = torch.Tensor(inputs)
    viewdirs_torch = torch.Tensor(viewdirs)

    ###################################

    # tf
    from run_nerf_helpers import init_nerf_model, get_embedder
    from run_nerf import run_network

    # Init
    embed_fn, input_ch = get_embedder(multires, i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(multires_views, i_embed)
    output_ch = 5 if N_importance > 0 else 4
    skips = [4]

    model_tf = init_nerf_model(input_ch=input_ch, output_ch=output_ch, skips=skips,
                               input_ch_views=input_ch_views, use_viewdirs=use_viewdirs)
    weights = model_tf.get_weights()

    with tf.GradientTape() as tape:
        # Forward pass
        outputs_tf = run_network(inputs_tf, viewdirs_tf, model_tf, embed_fn, embeddirs_fn, netchunk=1024*64)
        grad_vars = model_tf.trainable_variables
        loss_tf = tf.reduce_mean(tf.square(tf.zeros_like(outputs_tf) - outputs_tf))
        
        # Backward pass
        grads_tf = tape.gradient(loss_tf, grad_vars)  # same size as weights

    ###################################

    # torch
    from run_nerf_helpers_torch import NeRF
    from run_nerf_helpers_torch import get_embedder as get_embedder_torch
    from run_nerf_torch import run_network as run_network_torch

    # Init
    embed_fn, input_ch = get_embedder_torch(multires, i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder_torch(multires_views, i_embed)
    output_ch = 5 if N_importance > 0 else 4
    skips = [4] 

    model_torch = NeRF(input_ch=input_ch, output_ch=output_ch, skips=skips,
                       input_ch_views=input_ch_views, use_viewdirs=use_viewdirs)
    model_torch.load_weights_from_keras(weights)

    # Forward pass
    outputs_torch = run_network_torch(inputs_torch, viewdirs_torch, model_torch, embed_fn, embeddirs_fn, netchunk=1024*64)
    
    # Backward pass
    loss_torch = torch.mean((torch.zeros_like(outputs_torch) - outputs_torch) ** 2)
    loss_torch.backward()
    
    ###################################

    # Check outputs are the same
    assert np.allclose(outputs_tf.numpy(), outputs_torch.detach().numpy(), atol=1e-6)
    # Check first layer's gradients are the same
    assert np.allclose(np.transpose(grads_tf[0].numpy()), model_torch.pts_linears[0].weight.grad.numpy(), atol=1e-6)  
    

def test_load_blender_data():
    from load_blender_torch import load_blender_data as load_blender_data_torch
    from load_blender import load_blender_data as load_blender_data_tf

    datadir = './test_data/nerf_synthetic/lego'
    half_res = True
    testskip = 1
    white_bkgd = True

    images_torch, poses_torch, render_poses_torch, hwf_torch, i_split_torch = load_blender_data_torch(datadir, half_res, testskip)
    if white_bkgd:
        images_torch = images_torch[...,:3]*images_torch[...,-1:] + (1.-images_torch[...,-1:])
    else:
        images_torch = images_torch[...,:3]

    images_tf, poses_tf, render_poses_tf, hwf_tf, i_split_tf = load_blender_data_tf(datadir, half_res, testskip)
    if white_bkgd:
        images_tf = images_tf[...,:3]*images_tf[...,-1:] + (1.-images_tf[...,-1:])
    else:
        images_tf = images_tf[...,:3]

    assert np.allclose(images_torch, images_tf)
    assert np.allclose(poses_torch, poses_tf)
    assert np.allclose(render_poses_torch.numpy(), render_poses_tf.numpy())


def test_raw2outputs():
    raw_noise_std = 1
    white_bkgd = False

    raw = np.random.rand(10, 5, 4)
    z_vals = np.random.rand(10, 5)
    rays_d = np.random.rand(10, 3)

    raw_tf = tf.cast(raw, tf.float32)
    z_vals_tf = tf.cast(z_vals, tf.float32)
    rays_d_tf = tf.cast(rays_d, tf.float32)

    raw_torch = torch.Tensor(raw)
    z_vals_torch = torch.Tensor(z_vals)
    rays_d_torch = torch.Tensor(rays_d)

    from run_nerf_torch import raw2outputs as raw2outputs_torch
    # Function copied from run_nerf.py
    def raw2outputs_tf(raw, z_vals, rays_d, pytest=False):
        raw2alpha = lambda raw, dists, act_fn=tf.nn.relu: 1.-tf.exp(-act_fn(raw)*dists)
        
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = tf.concat([dists, tf.broadcast_to([1e10], dists[...,:1].shape)], -1) # [N_rays, N_samples]
        
        dists = dists * tf.linalg.norm(rays_d[...,None,:], axis=-1)

        rgb = tf.math.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[...,3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*noise.get_shape().as_list()) * raw_noise_std
                noise = tf.cast(noise, tf.float32)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        
        depth_map = tf.reduce_sum(weights * z_vals, -1) 
        disp_map = 1./tf.maximum(1e-10, depth_map / tf.reduce_sum(weights, -1))
        acc_map = tf.reduce_sum(weights, -1)
        
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map
    
    rgb_map_tf, disp_map_tf, acc_map_tf, weights_tf, depth_map_tf = raw2outputs_tf(raw_tf, z_vals_tf, rays_d_tf, pytest=True)
    rgb_map_torch, disp_map_torch, acc_map_torch, weights_torch, depth_map_torch = raw2outputs_torch(raw_torch, z_vals_torch, rays_d_torch, raw_noise_std, white_bkgd, pytest=True)
    
    assert np.allclose(rgb_map_tf.numpy(), rgb_map_torch.numpy())
    assert np.allclose(disp_map_tf.numpy(), disp_map_torch.numpy())
    assert np.allclose(acc_map_tf.numpy(), acc_map_torch.numpy())
    assert np.allclose(weights_tf.numpy(), weights_torch.numpy())
    assert np.allclose(depth_map_tf.numpy(), depth_map_torch.numpy())


def prepare_model():
    # Hyperparams
    use_viewdirs = True
    multires = 10
    multires_views = 4
    i_embed = 0
    N_importance = 64
    output_ch = 5 if N_importance > 0 else 4
    skips = [4]

    ###################################

    # tf
    from run_nerf_helpers import init_nerf_model
    from run_nerf_helpers import get_embedder as get_embedder_tf
    from run_nerf import run_network as run_network_tf

    # Init
    embed_fn_tf, input_ch = get_embedder_tf(multires, i_embed)
    input_ch_views = 0
    embeddirs_fn_tf = None
    if use_viewdirs:
        embeddirs_fn_tf, input_ch_views = get_embedder_tf(multires_views, i_embed)

    model_coarse_tf = init_nerf_model(input_ch=input_ch, output_ch=output_ch, skips=skips,
                                      input_ch_views=input_ch_views, use_viewdirs=use_viewdirs)
    model_fine_tf = init_nerf_model(input_ch=input_ch, output_ch=output_ch, skips=skips,
                                    input_ch_views=input_ch_views, use_viewdirs=use_viewdirs)
    network_query_fn_tf = lambda inputs, viewdirs, network_fn : run_network_tf(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn_tf, 
                                                                embeddirs_fn=embeddirs_fn_tf,
                                                                netchunk=1024*64)

    kwargs_tf = {'verbose': True, 
                 'retraw': True, 
                 'network_query_fn': network_query_fn_tf, 
                 'perturb': 1, 
                 'N_importance': 5, 
                 'network_fine': model_fine_tf, 
                 'N_samples': 5, 
                 'network_fn': model_coarse_tf, 
                 'white_bkgd': False, 
                 'raw_noise_std': 1.0,
                 'pytest': True}

    ###################################

    # torch
    from run_nerf_helpers_torch import NeRF
    from run_nerf_helpers_torch import get_embedder as get_embedder_torch
    from run_nerf_torch import run_network as run_network_torch
    
    # Init
    embed_fn_torch, input_ch = get_embedder_torch(multires, i_embed)
    input_ch_views = 0
    embeddirs_fn_torch = None
    if use_viewdirs:
        embeddirs_fn_torch, input_ch_views = get_embedder_torch(multires_views, i_embed)

    model_coarse_torch = NeRF(input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=use_viewdirs)
    model_coarse_torch.load_weights_from_keras(model_coarse_tf.get_weights())
    
    model_fine_torch = NeRF(input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=use_viewdirs)
    model_fine_torch.load_weights_from_keras(model_fine_tf.get_weights())

    network_query_fn_torch = lambda inputs, viewdirs, network_fn : run_network_torch(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn_torch, 
                                                                embeddirs_fn=embeddirs_fn_torch,
                                                                netchunk=1024*64)

    kwargs_torch = {'verbose': True, 
                    'retraw': True, 
                    'network_query_fn': network_query_fn_torch, 
                    'perturb': 1, 
                    'N_importance': 5, 
                    'network_fine': model_fine_torch, 
                    'N_samples': 5, 
                    'network_fn': model_coarse_torch, 
                    'white_bkgd': False, 
                    'raw_noise_std': 1.0,
                    'pytest': True}

    return kwargs_tf, kwargs_torch


def test_render_rays():
    from run_nerf import render_rays as render_rays_tf
    from run_nerf_torch import render_rays as render_rays_torch

    # Prepare data
    ray_batch = np.random.rand(10, 11)  # (batch, dim)
    ray_batch_tf = tf.cast(ray_batch, tf.float32)
    ray_batch_torch = torch.Tensor(ray_batch)

    kwargs_tf, kwargs_torch = prepare_model()

    # Run
    ret_tf = render_rays_tf(ray_batch_tf, **kwargs_tf)
    ret_torch = render_rays_torch(ray_batch_torch, **kwargs_torch)

    ###################################

    keys = ['rgb_map', 'disp_map', 'acc_map', 'raw']
    for key in keys:
        if key == 'raw':
            assert np.allclose(ret_torch[key].detach().numpy(), ret_tf[key].numpy(), atol=1e-5)
        else:
            assert np.allclose(ret_torch[key].detach().numpy(), ret_tf[key].numpy())


def test_render():    
    from run_nerf import render as render_tf
    from run_nerf_torch import render as render_torch
    
    # Prepare data
    H, W, focal = int(378/18), int(504/18), 407.5658/18
    chunk = 1024 * 32
    pose = np.array([
        [-0.9305, 0.1170, -0.3469, -1.3986],
        [-0.3661, -0.2975, 0.8817, 3.554],
        [0, 0.9475, 0.3197, 1.288]
    ])
    pose_tf = tf.cast(pose, tf.float32)
    pose_torch = torch.Tensor(pose)

    kwargs_tf, kwargs_torch = prepare_model()

    ret_tf = render_tf(H, W, focal, chunk=1024, c2w=pose_tf[:3,:4], use_viewdirs=True, **kwargs_tf)
    ret_torch = render_torch(H, W, focal, chunk=1024, c2w=pose_torch[:3, :4], use_viewdirs=True, **kwargs_torch)
    
    assert np.allclose(ret_tf[0].numpy(), ret_torch[0].detach().numpy(), atol=1e-4)
    assert np.allclose(ret_tf[2].numpy(), ret_torch[2].detach().numpy(), atol=1e-4)
    assert np.allclose(ret_tf[3]['rgb0'].numpy(), ret_torch[3]['rgb0'].detach().numpy(), atol=1e-4)
    """
    for i in range(len(ret_tf) - 1):
        print(i)
        assert np.allclose(ret_tf[i].numpy(), ret_torch[i].detach().numpy(), atol=1e-4)
    """

test_render()