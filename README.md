# VPCalibration
This project finds the focal length of the camera used to take an input image. Since the algorithm assumes the Manhattan-world, the input image must contain three group of lines that are mutually perpendicular. The project used T-linkage clustering to group the lines. Since the project is implemented to use nVidia GPU using CUDA for fast computation, you must have nVidia GPU to run this code.

## Prerequisites
The algorithm requires CUDA and OpenCV3.X. If you use OpenCV2.X, then you should change some code for finding lines.

## Compilation
This project is implemented to be compatible with both Linux and Windows. The detailed instruction will be updated soon.
