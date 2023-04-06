![图 1](../images/37d2b1e16366c723f6c06a57747980ef0d5f839a22cb4f5a0bbbcc5ab5f2cb56.png)  

- focal length 是在world units？
- N是什么意思？batch?
- principal point是什么？
- K是标定矩阵?相机的内参？
- NDC space是什么？screen space是什么？

数据集：
- bd是什么
- poses_arr的shape是什么意思？poses_arr被分成poses和bds



def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

imgs = [imread(f)[...,:3]/255. for f in imgfiles]

这样有意义吗？


def normalize(x):
    return x / np.linalg.norm(x)

这归一化有什么意义？