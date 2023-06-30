## precrop
- Seems not converge on Lego
- all white or black

<https://github.com/yenchenlin/nerf-pytorch/issues/5>

solution is `precrop` for
```
precrop_iters = 500
precrop_frac = 0.5
```