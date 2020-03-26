---
title: OpenACC CUDA Vector Add
author: GPU Teaching Kit -- Accelerated Computing
module: 21
---

# Objective

Implement a vector addition using OpenACC directives.

# Local Setup Instructions

The most recent version of source code for this lab along with the build-scripts can be found on the [Bitbucket repository](LINKTOLAB). A description on how to use the [CMake](https://cmake.org/) tool in along with how to build the labs for local development found in the [README](LINKTOREADME) document in the root of the repository.

The executable generated as a result of compiling the lab can be run using the following command:

```{.bash}
./OpenAccVectorAdd_Template -e <expected.raw> \
  -i <input0.raw>,<input1.raw> -o <output.raw> -t vector
```

where `<expected.raw>` is the expected output, `<input0.raw>,<input1.raw>` is the input dataset, and `<output.raw>` is an optional path to store the results. The datasets can be generated using the dataset generator built as part of the compilation process.

# Local Development & Obtaining a PGI Compiler License

The usage of OpenACC directives requires access to the PGI OpenACC compiler. Please follow the instructions on [Bitbucket repository](https://bitbucket.org/hwuligans/gputeachingkit-labs/src/master/Module23/OpenACCVectorAdd/PGI_LICENCE_INFO.markdown) to download the tools, generate the license file and install the license.
