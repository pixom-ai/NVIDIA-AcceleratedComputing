---
title: Vector Addition Using CUDA Streams
author: GPU Teaching Kit -- Accelerated Computing
module: 14
---

# Objective

The purpose of this lab is to get you familiar with using the CUDA streaming API by re-implementing a the vector addition lab to use CUDA streams.

# Prerequisites

Before starting this lab, make sure that:

- You have completed the vector addition lab
- You have completed all Module 14 lecture videos

# Instruction

Edit the code in the code tab to perform the following:

- You MUST use at least 4 CUDA streams in your program, but you may adjust it to be larger for largest datasets.
- Allocate device memory
- Interleave the host memory copy to device to hide 
- Initialize thread block and kernel grid dimensions
- Invoke CUDA kernel
- Copy results from device to host asynchronously

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```
./VectorAdd_Stream_Template -e <expected.raw> -i <intput1.raw>,<input2.raw> \
  -o <output.raw> -t vector
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.
