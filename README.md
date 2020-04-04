# NeRF-pytorch

A faithful PyTorch implementation of [NeRF](http://www.matthewtancik.com/nerf) that **reproduces** the results while running **1.3 times faster**. This repository is based on authors' Tensorflow implementation [here](https://github.com/bmild/nerf).

## Installation

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
python setup.py install
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - PyTorch 1.4
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse
  
The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.
  
</details>

## Run
To optimize a low-res Fern NeRF:
```
python run_nerf_torch.py --config config_fern.txt
```

After 200k iterations on a single 2080 Ti (about 8 hours)

## Method
<img src='imgs/pipeline.jpg'/>

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.

## Reproducing 

To make sure the implementation matches the official implementation, one can:

```
git checkout reproduce
py.test
```

---




## Running code

Here we show how to run our code on two example scenes. You can download the rest of the synthetic and real data used in the paper [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

### Optimizing a NeRF

Run
```
bash download_example_data.sh
```
to get the our synthetic Lego dataset and the LLFF Fern dataset.

To optimize a low-res Fern NeRF:
```
python run_nerf.py --config config_fern.txt
```
After 200k iterations (about 15 hours), you should get a video like this at `logs/fern_test/fern_test_spiral_200000_rgb.mp4`:

![ferngif](https://people.eecs.berkeley.edu/~bmild/nerf/fern_200k_256w.gif)

To optimize a low-res Lego NeRF:
```
python run_nerf.py --config config_lego.txt
```
After 200k iterations, you should get a video like this:

![legogif](https://people.eecs.berkeley.edu/~bmild/nerf/lego_200k_256w.gif)
