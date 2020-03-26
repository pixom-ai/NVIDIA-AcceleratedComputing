---
title: CUDA Image Color to Grayscale
author: GPU Teaching Kit -- Accelerated Computing
module: 3
---

# Objective

The purpose of this lab is to convert an RGB image into a gray scale image. The input is an RGB triple of float values and the student will convert that triple to a single float grayscale intensity value. A pseudo-code version of the algorithm is shown bellow:

```{.ruby}
for ii from 0 to height do
    for jj from 0 to width do
        idx = ii * width + jj
        # here channels is 3
        r = input[3*idx]
        g = input[3*idx + 1]
        b = input[3*idx + 2]
        grayImage[idx] = (0.21*r + 0.71*g + 0.07*b)
    end
end
```

# Prerequisites

Before starting this lab, make sure that:

- You have completed the required module videos

# Image Format

For people who are developing on their own system, the input image is stored in PPM `P6` format while the output grayscale image is stored in PPM `P5` format. Students can  create their own input images by exporting their image into PPM images. The easiest way to create image is via external tools. On Unix, `bmptoppm` converts BMP images to PPM images.

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
./ImageColorToGrayscale_Template -e <expected.pbm> \
    -i <input.ppm> -o <output.pbm> -t image`.
```

where `<expected.pbm>` is the expected output, `<input.ppm>` is the input dataset, and `<output.pbm>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.
