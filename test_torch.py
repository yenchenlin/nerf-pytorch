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

"""
def test_hierarchical_sampling():
    from run_nerf_helpers_torch import sample_pdf as sample_pdf_torch
    from run_nerf_helpers import sample_pdf as sample_pdf_tf

    bins = np.array([0., 1., 2., 4.])
    weights = np.array([1.0, 1.0, 1.0])
    N_samples = 3

    print(sample_pdf_tf(bins, weights, N_samples))



def test_model_architecture():
    from run_nerf_helpers_torch import init_nerf_model as init_nerf_model_torch
    from run_nerf_helpers import init_nerf_model as init_nerf_model_tf
"""


def test_load_blender_data():
    from load_blender_torch import load_blender_data as load_blender_data_torch
    from load_blender import load_blender_data as load_blender_data_tf

    datadir = './test_data/nerf_synthetic/lego'
    half_res = True
    testskip = 100
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


