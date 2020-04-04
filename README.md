# NeRF-pytorch

A faithful PyTorch implementation of [NeRF](http://www.matthewtancik.com/nerf) that **reproduces** the results while running **1.3 times faster**. This repository is based on authors' Tensorflow implementation [here](https://github.com/bmild/nerf).

## Dependencies:

- PyTorch 1.4
- matplotlib
- numpy
- imageio
- imageio-ffmpeg
- configargparse

The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.

## Installation

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
python setup.py install
```

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

### Rendering a NeRF

Run
```
bash download_example_weights.sh
```
to get a pretrained high-res NeRF for the Fern dataset. Now you can use [`render_demo.ipynb`](https://github.com/bmild/nerf/blob/master/render_demo.ipynb) to render new views.


### Extracting geometry from a NeRF

Check out [`extract_mesh.ipynb`](https://github.com/bmild/nerf/blob/master/extract_mesh.ipynb) for an example of running marching cubes to extract a triangle mesh from a trained NeRF network. Additional requirements: you'll need the [PyMCubes](https://github.com/pmneila/PyMCubes) package for marching cubes plus the [trimesh](https://github.com/mikedh/trimesh) and [pyrender](https://github.com/mmatl/pyrender) packages if you want to render the mesh inside the notebook.

## Generating poses for your own scenes

### Don't have poses?

We recommend using the `imgs2poses.py` script from the [LLFF code](https://github.com/fyusion/llff). Then you can pass the base scene directory into our code using `--datadir <myscene>` along with `-dataset_type llff`. You can take a look at the `config_fern.txt` config file for example settings to use for a forward facing scene. For a spherically captured 360 scene, we recomment adding the `--no_ndc --spherify --lindisp` flags.

### Already have poses!

In `run_nerf.py` and all other code, we use the same pose coordinate system as in OpenGL: the local camera coordinate system of an image is defined in a way that the X axis points to the right, the Y axis upwards, and the Z axis backwards as seen from the image.

Poses are stored as 3x4 numpy arrays that represent camera-to-world transformation matrices. The other data you will need is simple pinhole camera intrinsics (`hwf = [height, width, focal length]`) and near/far scene bounds. Take a look at [our data loading code](https://github.com/bmild/nerf/blob/master/run_nerf.py#L406) to see more.
