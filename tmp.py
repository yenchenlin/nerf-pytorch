import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
    
    relu = tf.keras.layers.ReLU()    
    dense = lambda W, act=relu : tf.keras.layers.Dense(W, activation=act)
    
    print('MODEL', input_ch, input_ch_views, type(input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)
    
    inputs = tf.keras.Input(shape=(input_ch + input_ch_views)) 
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])
    
    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)
                        
    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat([bottleneck, inputs_views], -1)  # concat viewdirs
        for i in range(D//2):
            outputs = dense(W//2)(inputs_viewdirs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = init_nerf_model(use_viewdirs=True)
print(model.summary())

print("--- Pytorch ---")

from run_nerf_helpers_torch import NeRF
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
model_torch = NeRF(use_viewdirs=True)
print(model_torch)
print(f"Number of parameters: {count_parameters(model_torch)}")
print(type(list(model_torch.parameters())))