- [1. configs](#1-configs)
  - [1.1. Parts](#11-parts)
    - [1.1.1. Logs](#111-logs)
    - [1.1.2. Dataset](#112-dataset)
    - [1.1.3. MLP](#113-mlp)
    - [1.1.4. Integration](#114-integration)
    - [1.1.5. Iterations](#115-iterations)

---
# 1. configs

## 1.1. Parts

### 1.1.1. Logs

![图 1](../images/de3f5daeb08d3a2541193c204ea9dce84d2c9a87f5051927185ab75b7dd00a46.png)  

### 1.1.2. Dataset

![图 2](../images/401ebbe044c52ad8afc7e2cf5e2382eae63d2ad9effac305666f434552b5fa7e.png)  

### 1.1.3. MLP

![图 3](../images/fbfc884acb13f75a226527b06cefb3147aac1dfca40b8c0271e7eeea5e4c5548.png)  

### 1.1.4. Integration

![图 4](../images/9904ff7aab2003cccf76a87e37a673e63d2ee3eff37780c149cc64d8436c2e40.png)  

### 1.1.5. Iterations

![图 5](../images/327f572110e7a25d4a87bd7ffc5548dda4be85aea371aa5c2246e9574b3793d1.png)  

> ckpt

`ft_path`: 指定加载，或者默认加载文件夹下最近的一次ckpt。

`no_reload`: 禁止默认load ckpt, 强制从头训练。

> quick start

```bash
ft_path = /path/to/trained.pt
render_only
# render_test
```

> render_poses

`render_poses` 本是从 `load_blender_data()` 返回的 spiral_poses，没有对应图片真值。
如果 `--render_test` ，那么 `render_poses` 就是测试集里的poses，有对应图片真值。

`i_video` 用的是 `render_poses`， `i_testset` 用的是 `poses[i_test]`