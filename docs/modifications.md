## outputch

<https://github.com/yenchenlin/nerf-pytorch/issues/22>

> yenchenlin `output_ch = 5 if args.N_importance > 0 else 4` should be 4, but he didn't modiify the code.

`output_ch = 4` 只在不使用方向时`use_viewdirs=False`有效果，正常使用方向时就是4. 所以使用不使用 `use_viewdirs` 都是4.