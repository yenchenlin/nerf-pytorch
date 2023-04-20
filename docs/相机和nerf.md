- [1. 物像关系](#1-物像关系)
- [2. FOV](#2-fov)
- [3. 相机坐标](#3-相机坐标)
  - [3.1. 内参](#31-内参)
  - [3.2. 外参](#32-外参)
- [4. 坐标系](#4-坐标系)

---
## 1. 物像关系

![图 16](../images/b3907e4ae3b5cf7e2ad5b2f5a29ad448f277b5479e20c1b95cd299c2ce656126.png)  

![图 17](../images/df934186093fafd9f9590af9a0e32b5c441cceda3cedc23f6ab242324051a987.png)  


$\dfrac{1}{f} = \dfrac{1}{z_0} + \dfrac{1}{z_i}$

f为焦距， $z_0$为物距（物体到透镜的距离）， $z_i$为像距（透镜到像面的距离）。


[由此可以推出不同的成像关系](https://www.zhihu.com/question/38929736/answer/2327108553)

## 2. FOV
视场 Field of view（FOV），是一个角度。

> 当传感器大小固定时，焦距越短，FOV越大；焦距越长，FOV越小。

![图 11](../images/ee747b21ed49adc9a8a76c96dfca5e6906b09dfed5bb16818d20f47af7f91a6d.png)

> 当焦距固定时，传感器大小越小，FOV越小。

![图 14](../images/cb9e61b95bb327af9712617afd7ac0de986308aeb7bc522376dddd29d8235b98.png)  


> 公式就是tan三角函数联系起焦距与高宽、或物距与物高宽。

![图 1](../images/42233c45b2aac9127c39b30870163a7a6cf81a6e7bb3da56bc8c1d73973a5155.png)  


![图 1](../images/2db836f5a66a8f1a2111baafb5742a7a286a6c8225ec8e6db47c299d391556fc.png)  

W是图像的宽度（传感器大小），$\omega$是x轴的视角宽度，f是焦距。
```python
# `camera_angle_x`即在x轴的视角，对应图像的宽度。
focal = .5 * W / np.tan(.5 * camera_angle_x)
```
`"camera_angle_x": 0.5235987755982988,` 就是比如30°， Π/6 = 0.5235987755982988

`W`的单位是pixel， f的单位也是pixel。


> 镜头参数

![图 12](../images/53da8b7cab4fd935b085033322fe5dd77aa8b274a90dabff368d2862238f5b09.png)  

我们以35mm-format的底片大小为标准，17mm、50mm、200mm、28mm的焦距是在这样大小的底片上，这个焦距是等效的虚指。意思是，实际手机镜头的焦距很小，对应的也是很小的底片大小。

拍照的效果就是，视场越窄，镜头拍到的就越远。

![图 13](../images/e2954d730381f66ae2b5dbef7c487b20f207d99d59d28ed24d7a42afe1fe965c.png)  

> maintain same FOV

![图 15](../images/0cb82c87d007dc87a7a461bd68515b29d8bd636eaf55de47619228a3af050c1c.png)  


## 3. 相机坐标

相机两个参数：内参和外参
- 内参intrinsics，即**K**
    固定不变的，如果不知道可以通过**标定**求得。
- 外参extrinsics，即**T**
    描述相机的位姿，即变换矩阵（Transform Matrix），由旋转R和平移t组成。
- 内参共有，外参变化：
    由于多组图片都是同一个相机拍摄得到，所以其内参数由多组图像共有，而外参数随着不同的图像将发生变化
- 都用齐次坐标表示
  

### 3.1. 内参

![图 1](../images/02b6dfb44b504e580ef2310b0d31a35d373315955bd015030b909f02a60930d5.png)  
![图 2](../images/96bc53ad80a74e6f18c6d1e53ff8c902ef7ff579dfcae343d9447b55513c83d7.png)  
![图 3](../images/114fc6203fd25de747d5f21f986407a23ab0b7b0147e94825892b53f7bf97b55.png)  

```python
# 平移到图像中心
K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]
])
```

### 3.2. 外参

相机外参是一个4x4的矩阵$T$，其作用是将世界坐标系的点$P_{world}=[x,y,z,1]$变换到相机坐标系下

$$P_{camera}=MP_{world}$$

我们也把相机外参叫做**world-to-camera (w2c)矩阵**。相机外参的逆矩阵被称为**camera-to-world (c2w)矩阵**。其作用是把相机坐标系的点变换到世界坐标系。

NeRF主要使用c2w，这里详细介绍一下c2w的含义。c2w矩阵是一个4x4的矩阵，左上角3x3是旋转矩阵R，又上角的3x1向量是平移向量T。有时写的时候可以忽略最后一行[0,0,0,1]。

![图 6](../images/1d0fc5c458d0b57f2cec5dc3607ddb3344d04b0477efe23591bb0b3a9a3283a2.png)  

- R
    $R \in SO(3)$
    $SO(n) = \{R \in \R^{n\times n}|RR^T=I,det(R)=1\}$
    - 旋转矩阵是一个正交矩阵, 正交矩阵的逆等于其转置矩阵。旋转矩阵的逆等于其转置矩阵
    - 行列式值为1

- M
    $M \in SE(3)$
    $SE(n) = \left\{T=\left[\begin{array}{cc}R & t\\ 0^T & 1\end{array}\right] \in \R^{4\times 4}|R\in SO(3),t\in \R^3\right\}$


![图 7](../images/c2b2c7aff71ab6c0053f2367b48b604c39093e0134a9f8d8f2b46afc01b6b0d0.png)  
旋转矩阵的第一列到第三列分别表示了相机坐标系的X, Y, Z轴在世界坐标系下对应的方向；平移向量表示的是相机原点在世界坐标系的对应位置。



## 4. 坐标系

右手法则，用右手的**4个指头从a转向b**，大拇指朝向就是aXb的方向。

![图 2](../images/6d063cd15878e415ddb39c6cd88bce1593b60e2376f06373b7aa7493f9b87758.png)  

> ABC对应XYZ

![图 8](../images/0b2e24a1c6d97650f49d5d02e08f0f244fa60e9a700c62330261ddb30daeb61d.png)  


![图 4](../images/f543774d2d57cb991abf472608bf10221a356f9e287b95aad9a49a5d7548670f.png)  

![图 3](../images/d7811eeb810841979e5f8cbd88f6e6d71e2744c4d464081863dd8d93079e2370.png)  


![图 4](../images/48b46404b809bc67b121791f53db54e68eb85eaec2c3968048c70024f1e81280.png)  
![图 5](../images/872d8ed4d1075b1b5cadc6434a6987e6d28d73e50a72fcdb8dc49a8cb743a7c6.png)  
