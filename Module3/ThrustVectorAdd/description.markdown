---
title: CUDA Thrust Vector Add
author: GPU Teaching Kit -- Accelerated Computing
module: 3
---

# Objective

The purpose of this lab is to introduce the student to the CUDA API by implementing vector addition using Thrust.

# Prerequisites

Before starting this lab, make sure that:

- You have completed all of Module 2 in the teaching kit
- You have completed the "Device Query" lab

# Instructions

Edit the code in the code tab to perform the following:

- Generate a `thrust::dev_ptr<float>` for host input arrays
- Copy host memory to device
- Invoke `thrust::transform()`
- Copy results from device to host

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```
./ThrustVectorAdd_Template -e <expected.raw> \
  -i <input0.raw>,<input1.raw> -o <output.raw> -t vector
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.
