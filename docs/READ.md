![图 1](../images/37d2b1e16366c723f6c06a57747980ef0d5f839a22cb4f5a0bbbcc5ab5f2cb56.png)  

- focal length 是在world units？
- N是什么意思？batch?
- principal point是什么？
- NDC space是什么？screen space是什么？




def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

imgs = [imread(f)[...,:3]/255. for f in imgfiles]



