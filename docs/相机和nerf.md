[NeRF代码解读-相机参数与坐标系变换](https://zhuanlan.zhihu.com/p/593204605)

## 物像关系

$\dfrac{1}{f} = \dfrac{1}{d_1} + \dfrac{1}{d_2}$

f为焦距，d1为物距（物体到透镜的距离），d2为像距（透镜到像面的距离）。

[由此可以推出不同的成像关系](https://www.zhihu.com/question/38929736/answer/2327108553)

## FOV
视场角，即Field of view（FOV）。焦距越短，FOV越大；焦距越长，FOV越小。

公式就是tan三角函数联系起焦距与高宽、或物距与物高宽。

![图 1](../images/42233c45b2aac9127c39b30870163a7a6cf81a6e7bb3da56bc8c1d73973a5155.png)  


![图 1](../images/2db836f5a66a8f1a2111baafb5742a7a286a6c8225ec8e6db47c299d391556fc.png)  

  W是图像的宽度，$\omega$是x轴的视角宽度，f是焦距。
```python
# `camera_angle_x`即在x轴的视角，对应图像的宽度。
focal = .5 * W / np.tan(.5 * camera_angle_x)

```
`"camera_angle_x": 0.5235987755982988,` 就是比如30°， Π/6 = 0.5235987755982988

`W`的单位是pixel， f的单位也是pixel。

## 相机坐标

相机两个参数：内参和外参
- 内参intrinsics，即**K**
    固定不变的，如果不知道可以通过**标定**求得。
- 外参extrinsics，即**T**
    描述相机的位姿，即变换矩阵（Transform Matrix），由旋转R和平移t组成。
- 内参共有，外参变化：
    由于多组图片都是同一个相机拍摄得到，所以其内参数由多组图像共有，而外参数随着不同的图像将发生变化
- 都用齐次坐标表示
  

内参：

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

外参：
相机外参是一个4x4的矩阵$T$，其作用是将世界坐标系的点$P_{world}=[x,y,z,1]$变换到相机坐标系下$P_{camera}=TP_{world}$。我们也把相机外参叫做**world-to-camera (w2c)矩阵**。相机外参的逆矩阵被称为**camera-to-world (c2w)矩阵**。其作用是把相机坐标系的点变换到世界坐标系。

NeRF主要使用c2w，这里详细介绍一下c2w的含义。c2w矩阵是一个4x4的矩阵，左上角3x3是旋转矩阵R，又上角的3x1向量是平移向量T。有时写的时候可以忽略最后一行[0,0,0,1]。



![图 6](../images/1d0fc5c458d0b57f2cec5dc3607ddb3344d04b0477efe23591bb0b3a9a3283a2.png)  


![图 7](../images/c2b2c7aff71ab6c0053f2367b48b604c39093e0134a9f8d8f2b46afc01b6b0d0.png)  
旋转矩阵的第一列到第三列分别表示了相机坐标系的X, Y, Z轴在世界坐标系下对应的方向；平移向量表示的是相机原点在世界坐标系的对应位置。

## 坐标系

世界坐标系，相机坐标系，像素坐标系

![图 4](../images/f543774d2d57cb991abf472608bf10221a356f9e287b95aad9a49a5d7548670f.png)  

> X轴指向right，Y轴指向Down，Z轴指向Foward

![图 5](../images/171650a86fa2620789dad672d5f50e3c45cafe8dc748c840b9464d481b48545b.png)  

![图 8](../images/2c5ea1444f1a3c889a7031adb86d496845824931637e11f01f3efd54853b303c.png)  
