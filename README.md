This repo contains code to process and render images of the MIT Pushing dataset.

First, create a folder structure like [pushdata folder]/[surface name]/. 
Below we will use ~/pd as the pushdata root folder, and abs as the surface.

Download the zip files of h5 format from mcube.mit.edu/push-dataset to the folders.
I.e. The data of rect1 on ABS surface will be at ~/pd/abs/rect1_h5.zip.

First preprocess the data using 
```
scripts/preprocess.py --source-dir ~/pd
```
This will
  * remove redundant data entries 
  * treat some (not all) of the jumps in object orientation
  * transform the orientation to [-pi, pi]
  * synchronize the data by resampling to a given frequency
  * set the initial object position and orientation to zero
  * add information about the push (angle, velocity...) to the h5 data files

To render RGB-D pictures from the preprocessed images, refer to 
```scripts/render_scene.py```. 

Assuming we are at the top folder of pdproc, a rendering demo can be run by
```
scripts/render_scene.py --source-folder ~/pd --out-dir .
```
This will generate a series of images in jpeg in the current folder.

To annotate contact modes, please refer to ```scripts/render_scene.py```
