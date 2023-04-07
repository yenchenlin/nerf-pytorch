# load_blender.py


`load_blender_data()`

```json
{
    "camera_angle_x": 0.6911112070083618,
    "frames": [
        {
            "file_path": "./train/r_0",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    -0.9250140190124512,
                    0.2748899757862091,
                    -0.2622683644294739,
                    -1.0572376251220703
                ],
                [
                    -0.3799331784248352,
                    -0.6692678928375244,
                    0.6385383605957031,
                    2.5740303993225098
                ],
                [
                    0.0,
                    0.6903012990951538,
                    0.7235219478607178,
                    2.9166102409362793
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
```
- `camera_angle_x`: 算内参的焦距 focal 时用到的FOV
- `rotation`: 没用到。
- `transform_matrix`: 外参之c2w。`load_blender_data`返回的`poses`, `get_rays`的`c2w`。

```python
# 根据此划分
splits = ['train', 'val', 'test']
# 划分下标
# [ [ 0,  1, ..., 99], [100, 101, ..., 112], [113, 114, ..., 137] ]
i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
# run_nerf中
i_train, i_val, i_test = i_split


# (138, 800, 800, 4) (index, H, W, C), C是RGBA
imgs = np.concatenate(all_imgs, 0)
# (138, 4, 4), json中的transform_matrix，外参T
poses = np.concatenate(all_poses, 0)


# scalar, 1111.1110311937682
focal = .5 * W / np.tan(.5 * camera_angle_x)

# (40, 4, 4)
render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

```
# run_nerf.py

`train()`

```python
if args.white_bkgd:
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
else:
    images = images[...,:3]
```

```python
if K is None:
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
```



# run_nerf_helpers.py

```python
class NeRF(nn.Module):
    def __init__(self, 
        D=8,                    # 默认8层MLP，对应 args.netdepth 和 args.netdepth_fine
        W=256,                  # 默认每层256，对应 args.netwidth 和 args.netwidth_fine
        input_ch=3,             # x,y,z
        input_ch_views=3,       # direction的三维表示
        output_ch=4,            # rgb和sigma
        skips=[4],              # 残差
        use_viewdirs=False 
    ):
```

# run_nerf.py

`create_nerf()`
```python
# 位置编码 location，由3维变成63维度。
embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
# args.multires = 10, sin和cos几次。
# args.i_embed = 0， 默认表示使用编码。
# input_ch = 63， 63 = 1 * 3 + （2 * 10）* 3 = 原本xyz + （sin, cos 10次）* 3维度
    # def get_embedder(multires, i=0):
    #     if i == -1:
    #         return nn.Identity(), 3
        
    #     embed_kwargs = {
    #                 'include_input' : True,       # 表示加入原生的与否，这就是同原版position encoding的不同之处。
    #                 'input_dims' : 3,
    #                 'max_freq_log2' : multires-1, # 9
    #                 'num_freqs' : multires,       # 10, sin和cos几次。
    #                 'log_sampling' : True,        # 决定是1, 2, 4, 8 ,16 还是 1到16等距离采样。
    #                 'periodic_fns' : [torch.sin, torch.cos],
    #     }
        
    #     embedder_obj = Embedder(**embed_kwargs)
    #     embed = lambda x, eo=embedder_obj : eo.embed(x)
    #     return embed, embedder_obj.out_dim

# 位置编码 direction，由3维变成27维度。
embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
# args.multires_views = 4, sin和cos几次。
# input_ch = 27， 27 = 1 * 3 + （2 * 4）* 3 = 原本diretion的三维度表示 + （sin, cos 4次）* 3维度
```

### MLP
![图 2](../images/093cbe95a5eabf5685a649913018b32ba5f5b14493e704cba43cca7aaa8b7cfb.png)  


```python
N_importance = 128
# parser.add_argument("--N_importance", type=int, default=0,
#                         help='number of additional fine samples per ray')

# 没什么意义，在NeRF mdoel定义中， output_ch 只在不使用 use_viewdirs 时才作用
output_ch = 5 if args.N_importance > 0 else 4


skips = [4] # 取默认值

### model是粗模型，model_fine是精细模型
model = NeRF(D=args.netdepth, W=args.netwidth,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

if args.N_importance > 0:
    model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                        input_ch=input_ch, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
```
![图 3](../images/f84bc392c86aad926acab0a8b5c976d2a58ff552053199b2a8d42e857084a475.png)  

```python
# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    
```

## train

```python

    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
```


```
# 因为最后一行都是`[0, 0, 0, 1]`
pose = poses[img_i, :3,:4]
```


## render

```python
def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
```