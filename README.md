# VPCalibration
This project finds the focal length of the camera used to take an input image. Since the algorithm assumes the Manhattan-world, the input image must contain three group of lines that are mutually perpendicular. The project used T-linkage clustering to group the lines. Since the project is implemented to use nVidia GPU using CUDA for fast computation, you must have nVidia GPU to run this code.

## Prerequisites
This algorithm depends on the following libraries.
- CUDA
- OpenCV 3.X Packages include
  - core, highgui, imgcodecs, imgproc, calib3d, and line_descriptor of extra module.
- Ceres-Solver

Since the algorithm depends on CUDA, you must have nVidia GPU that supports the CUDA programming. Also, you must setup the libraries and link them properly to run this program on Windows system with Visual Studio.

## Compilation
This project is implemented to be compatible with both Linux and Windows. The detailed instruction on Linux system will be updated soon.
