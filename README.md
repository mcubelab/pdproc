This repo contains code to process and render images of the MIT Pushing dataset (http://web.mit.edu/mcube//push-dataset/).

Preprocessing
=======
First, create a folder structure like [pushdata folder]/[surface name]/ by 
downloading the zip-files for each surface to [pushdata folder]/ and unzipping 
them there. This should result in a folder structure like this:
[pushdata folder]/[surface name]/[object name]/[object_name]_h5.zip

Below we will use \~/pd as the pushdata root folder.
For example, the data of rect1 on ABS surface will be at \~/pd/abs/rect1/rect1_h5.zip.

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

This will save all the preprocessed h5 files for object [object_name] on 
surface [surface_name] to \~/pd/[surface name]/[object name]. 
If you want to write output to another directory than the source directory 
(\~/pd), you can specify this with the --out-dir argument. 

RGB-D rendering
=======
The preprocessing has to be done before.

To render RGB-D pictures from the preprocessed images, refer to 
```scripts/render_scene.py```. 

Assuming we are at the top folder of pdproc, a rendering demo can be run by
```
scripts/render_scene.py --source-folder ~/pd --out-dir .
```
This will generate a series of images in jpeg in the current folder.

Contact annotation
=======
To annotate the preprocessed h5 files with contact points and surface normals, 
please refer to ```scripts/render_scene.py```.
