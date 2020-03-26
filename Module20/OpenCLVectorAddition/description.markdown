---
title: OpenCL Vector Addition
author: GPU Teaching Kit -- Accelerated Computing
module: 20
---

# Objective

The purpose of this lab is to introduce the student to the OpenCL API by implementing vector addition. The student will implement vector addition by writing the GPU kernel code as well as the associated host code.

# Prerequisites

Before starting this lab, make sure that:

- You have completed all of the module video lectures.
- You have completed the CUDA Vector Addition lab.

# Instructions

This lab uses a separate build system. Consult the provided `Makefile`.

- Edit the `Makefile` variable `LIBWB` to point to the location of `libwb.so`. If the Modules were built in `/path/to/build`, that location should be `/path/to/build`.
- Edit the `Makefile` target `all` to look like `all: template solution` to compile the solution.

Edit the code in the code tab to perform the following:

- Set up an OpenCL context and command queue
- Invoke the OpenCL API to build the kernel
- Allocate device memory
- Copy host memory to device
- Initialize work-group and global sizes
- Enqueue the kernel
- Copy results from device to host
- Free device memory
- Write the OpenCL kernel

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./OpenCLVectorAdd_Template -e <expected.raw> -i <intput1.raw>,<input2.raw> \
  -o <output.raw> -t vector
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.

The local CMake does not build this lab. An example `Makefile` is

```{.bash}
NVCC=nvcc
INCLUDE= -I../../libwb
LIBWB= -L../../build -lwb
LIBS= -lOpenCL $(LIBWB)

all: template

template:
    $(NVCC) -std=c++11 template.cpp $(INCLUDE) $(LIBS) -o OpenCLVectorAdd_Template

solution:
    $(NVCC) solution.c $(INCLUDE)

clean:
    rm -f OpenCLVectorAdd_Template
```
