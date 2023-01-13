# Depth-Based Image Reprojection (frame interpolation)
## Description
This project attempts to solve the following task:  
*Given two images taken from two different views of a scene and their accompanying 
depth maps, generate a novel view of the scene.*
  
For the purposes of this project, the images are assumed to be taken by a stereo
setup and rectified. The depth maps are assumed to be pixel disparity derived
from the stereo setup.  
  
Let's say the images were taken by two cameras: `camera0` and `camera1`. Let's define
a number `alpha` representing a point along the line connecting the two cameras,
such that `camera0` sits at `alpha=0` and `camera1` sits at `alpha=1`. A view 
halfway between these views would sit at `alpha=0.5`. The basis for this algorithm
is to use the disparity maps as a forward mapping from one image to the other, so by
multiplying the left disparity map by `alpha`, we obtain a mapping from the left 
image to image `alpha`. Similarly, we obtain an estimate of the same image by 
multiplying the right disparity by `(1 - alpha)`. Using these two estimates of the 
same image, we can merge them into a coherent result by resolving holes caused 
by occluded regions and removing artifacts.

## Algorithm
1. First, the images and disparity maps are processed to ensure they all have the 
same width and height. 
2. Then, perform a forward-warp horizontally in both directions. The left image 
is warped by `alpha * img0` and the right image is warped by 
`(1 - alpha) * img1`. The disparity maps themselves are also warped, providing a
depth estimate at the two new generated views.
3. Resolve holes caused by occluded regions. Since occluded regions in one view
are sometimes unoccluded in the other view, you can simply fill the holes by splatting
the pixels from the other warped image that are in the same location. The images
are also combined using the depth maps to determine which pixels to splat from each
image.
4. Fill any remaining holes via interpolation. First, the images are downsampled
in order to speed up interpolation. Then, Navier-Stokes inpainting is used to fill
any holes. The result is then scaled back up and holes are filled the same way as in
step (3).
5. Filter the image to remove minor artifacts. The splatting process can result
in minor artifacts spread throughout the image. Many of these artifacts are just 
a single pixel out of place. To resolve these artifacts, I use median blurring.

## Results
Some results of this algorithm can be found below; the two examples pan from `alpha=0` 
to `alpha=1`. It seems to perform well in general, but still suffers from some minor 
artifacts. From experimentation, these artifacts seem to be caused by slight vertical 
misalignment between the images (since the disparity maps represent purely horizontal 
disparity). One way to resolve these artifacts would be to use optical flow instead of 
the given disparity map in order to provide a bidrectional forward mapping, but this 
causes significant overhead for the algorithm (especially for high-resolution images). 
Additionally, this algorithm struggles with generating images where `alpha < 0` or 
`alpha > 1` since these tend to have significant holes caused by regions occluded in 
both warped images.  
![adirondack](examples/example1.png)
![motorcycle](examples/example2.png)

## Environment
I'm using a `python3.8` virtual environment. To run different versions of
python alongside each other in Ubuntu, I used the [deadsnakes
PPA](https://github.com/deadsnakes). A 
tutorial for Ubuntu can be found [here](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04).
Once installed, use the following steps:  
`$ python3.8 -m venv [name of environment]`  
`$ source [name of environment]/bin/activate`  
`$ pip install opencv-python matplotlib`  
Now you should be able to run `main.py` locally.

## Running the Code
To download some example images and disparity maps from the [Middlebury](https://vision.middlebury.edu/stereo/data/scenes2014/) 
dataset, run  
`$ ./download.sh`  
This will create a directory `images/` with three subdirectories `adirondack/`,
`jadeplant/`, and `motorcycle/`.  
  
Running the code will look something like:  
`(env) $ python main.py images/adirondack/ 0.5`  
The first argument is the path to an image directory that contains the four files
`im0.png`, `im1.png`, `disp0.pfm`, and `disp1.pfm`. The second argument is
the number `alpha` at which you'd like to generate a view (by default, `alpha=0.5`).  

Another option would be to run something like:  
`(env) $ python main.py images/adirondack/ -p 0 1 -s`  
The `-p` flag is optional; it allows you to generate 10 views panning from one
value of `alpha` to the other. The above command generates a gif panning from
`alpha=0` to `alpha=1`. The `-s` flag is also optional; if provided, it saves the
generated image/gif.
