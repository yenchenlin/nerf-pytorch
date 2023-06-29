## render_poses

`render_poses` 本是从 `load_blender_data()` 返回的 spiral_poses，没有对应图片真值。
如果 `--render_test` ，那么 `render_poses` 就是测试集里的poses，有对应图片真值。

`i_video` 用的是 `render_poses`， `i_testset` 用的是 `poses[i_test]`