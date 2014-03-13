atan_calib
==========

Calibration application for the arctangens lensmodel.

**WARNING** This program is supplied as is. It might not work for your camera, and the code may blow up unexpectedly.


Lens model
-------------------------------
The model is represented by three parameters:

*  `wc = [wx, wy]` is the radial center
* `lgamma` is the distortion parameter

Run the calibration
----------------------
Required packages: OpenCV, h5py, scipy, numpy, matplotlib

To calibrate, prepare a directory which contain images (currently hardcoded to only look for PNG images) depicting a standard chessboard calibration pattern.
Make sure the images are sharp and that the chessboard are positioned close to the edges (more distortion) for at least some of the images.

Run as

    python atan_calib.py /path/to/images --chessbord 6 6
    

How does it work
--------------------
1. First it will make a standard OpenCV camera calibration with all lens parameters set to zero.
2. The output camera matrix, and camera positions, is then used as start value for an optimizer that minimizes the projection errors given the lens distortion parameters (and optimized camera positions and camera matrix).
3. Output is the new camera matrix and lens distortion parameters.

How do I use the output parameters?
---------------------------------------
Look at the source code for the part that lets you show image rectification results for examples of how the lens model is applied.

Todo
-----
* Clean the code
* Describe model better, with equations and references.
