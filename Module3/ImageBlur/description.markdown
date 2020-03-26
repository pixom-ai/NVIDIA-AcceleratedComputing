---
title: CUDA Image Blur
author: GPU Teaching Kit -- Accelerated Computing
module: 3
---

# Objective

The purpose of this lab is to implement an efficient image blurring algorithm for an input image. Like the image convolution Lab, the image is represented as `RGB float` values. You will operate directly on the RGB float values and use a 3x3 Box Filter to blur the original image to produce the blurred image.

# Prerequisites

Before starting this lab, make sure that:

- You have completed all Module 3 lecture videos

# Instructions

Edit the code in the code tab to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./ImageBlur_Template -e <expected.ppm> -i <input.ppm> \
  -o <output.ppm> -t image
```

where `<expected.ppm>` is the expected output, `<input.ppm>` is the input dataset, and `<output.ppm>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.
