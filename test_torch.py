import pytest
import numpy as np
import torch
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def test_positional_encoding():
    from create_nerf_torch import get_embedder as get_embedder_torch
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


def test_model_architecture():
    from create_nerf_torch import init_nerf_model as init_nerf_model_torch
    from run_nerf_helpers import init_nerf_model as init_nerf_model_tf



    # TODO: add assertion
    # assert ???