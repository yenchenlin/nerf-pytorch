- [NeRF model](#nerf-model)
  - [create\_nerf](#create_nerf)
---



# NeRF model


首先 NeRF 将场景用 MLP 表示，使用坐标 $\boldsymbol{x}$ 推测出密度 $\sigma$ 和中间特征，然后用这个中间特征 $\boldsymbol{e}$ 和视角 $\boldsymbol{d}$ 推测出这个点的颜色 $\boldsymbol{c}$，下面将这两个过程分开写，其实就是 NeRF 中的网络：

$$
\begin{aligned} (\sigma, \boldsymbol{e}) &=\operatorname{MLP}^{(\mathrm{pos})}(\boldsymbol{x}), \\ \boldsymbol{c} &=\operatorname{MLP}^{(\mathrm{rgb})}(\boldsymbol{e}, \boldsymbol{d}) \end{aligned}\\
$$

![图 2](../images/093cbe95a5eabf5685a649913018b32ba5f5b14493e704cba43cca7aaa8b7cfb.png)  

MLP的架构，使得体积密度仅被预测为 **3D 位置**的函数，而发射的辐射率被预测为 **3D 位置和 2D 观察方向**的函数。

MLP 可以被认为是为每个输入的 3D 位置预测一个 256 维的**特征向量**，然后将其与观察方向连接起来并解码为**RGB 颜色**。


![图 8](../images/3ae31f519d1d3fc3ae0b5b24d8dbbf0af67c89b67b819024546d9b461caf5c23.png)  




The final activations in generating σ(t) and c(t) are a **ReLU** and a **sigmoid** respectively, as density must be non-negative and color must be in [0, 1]. (激活函数没有写在模型中，而是写在了积分公式处)

```python
class NeRF(nn.Module):
    def __init__(self, 
        D=8,                    # 默认8层MLP，对应 args.netdepth 和 args.netdepth_fine
        W=256,                  # 默认每层256，对应 args.netwidth 和 args.netwidth_fine
        input_ch=3,             # x,y,z
        input_ch_views=3,       # direction的三维表示
        output_ch=4,            # rgb和sigma
        skips=[4],              # skip connnect（不是残差）, 第六个MLP（不是第4，第5）
        use_viewdirs=False 
    ):
```

![图 11](../images/362834bb2aef579ae8ef60154b54951ba9058c268e9ba4fb6a3cbaf415fdccc0.png)  

> Paper: a skip connection that concatenates this input to the fifth layer’s activation

这个图的示意不是第五层的输入加入了x，而是x和第五层的输出concatenate起来给第六层。
## create_nerf

![图 4](../images/7efe5124adc88aee91b5e3d382362233b7b3b49de3b5d605e498a957937f4e2f.png)  

