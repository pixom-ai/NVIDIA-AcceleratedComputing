# Accelerated Computing Teaching Kit Lab/solution Repository

Developed by Abdul Dakkak and Carl Pearson

Welcome to the Accelerated Computing Teaching Kit Lab/solution repository. The kit and associated labs are produced jointly by NVIDIA and University of Illinois (UIUC).  All material is available under the [Creative Commons Attribution-NonCommercial License](http://creativecommons.org/licenses/by-nc/4.0/).
 
## System and Software Requirements

You must use an [NVIDIA CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus)
to use the compiled binaries. 

**Don't have access to GPUs? The GPU Teaching Kit comes with codes worth up to $125 of Amazon Web Services (AWS) GPU compute credit for each student in your course, as well as $200 for yourself as the instructor, to provide a GPU compute platform to work on the open-ended labs. Learn how to run these labs on AWS GPUs in the cloud using the [NVIDIA-Docker instructions](#markdown-header-using-docker).**

The labs in the Teaching Kit require a CUDA supported operating system,
C compiler, and the CUDA 8 Toolkit. The CUDA 8 Toolkit can be downloaded
from the [CUDA Download](https://developer.nvidia.com/cuda-downloads) page.
Instructions on how to install the CUDA Toolkit are available in the
[Quick Start page](http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
Installation guides and the list of supported C compilers for [Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html),
[Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
[OSX](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html) are
also found in the [CUDA Toolkit Documentation Page](http://docs.nvidia.com/cuda/index.html).

Aside from a C compiler and the CUDA Toolkit, [CMake](https://cmake.org/) 2.8 or later is required
to generate build scripts for your target IDE and compiler. 
  
**The usage of OpenACC directives for the Module 21 labs requires access to the PGI OpenACC compiler.**
Please follow the instructions [here](https://bitbucket.org/hwuligans/gputeachingkit-labs/src/master/Module21/OpenACCVectorAdd/PGI_LICENCE_INFO.markdown?fileviewer=file-view-default) in this Bitbucket repository to download the
tools, generate the license file and install the license.
 
The next section describes the process of compiling and running a lab.

## Compiling and Running Labs

In this section we describe how to setup your machine to compile the labs.
First, regardless of the platform compiling the labs the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and
[CMake](https://cmake.org/) must be installed.

Now, checkout the the GPU Teaching Kit -- Accelerated Computing Labs from this Bitbucket repository.

Since, the project depends on an external [libwb](https://github.com/abduld/libwb) repository [![Build Status](https://travis-ci.org/abduld/libwb.svg?branch=master)](https://travis-ci.org/abduld/libwb)
 we must perform a recursive clone (to also checkout the `libwb` repository).

~~~{.bash}
git clone --recursive git@bitbucket.org:hwuligans/gputeachingkit-labs.git
~~~

In the next section we will show how to compile and run the labs on Linux, OSX,
and Windows.

### Linux and Mac OSX

We will show how to compile the labs on both Linux and OSX using Makefiles.
First, create the target build directory

~~~
mkdir build-dir
cd build-dir
~~~

We will use `ccmake`

~~~
ccmake /path/to/gpu-kit-git-checkout
~~~

You will see the following screen

![ccmake](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+11.58.27.png)

Pressing `c` would configure the build to your system (in the process detecting
  the compiler, the CUDA Toolkit location, etc...).

![ccmake-config](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.03.26.png)

Note the options available to you, specifically:

~~~
BUILD_DESCRIPTION               *OFF
BUILD_DATASET_GENERATOR         *ON
BUILD_LIBWB_LIBRARY             *ON
BUILD_SOLUTION                  *ON
BUILD_TEMPLATE                  *OFF
~~~

* `BUILD_DESCRIPTION` -- option toggles whether to regenerate
`pdf` and `docx` lab output (this requires a python, latex, and pandoc installation)
* `BUILD_DATASET_GENERATOR` -- option toggles whether to build the dataset
generator scripts as part of the build process
* `BUILD_LIBWB_LIBRARY` -- option toggles whether to build the `libwb` (the support library)
as part of the build process
* `BUILD_TEMPLATE` -- option toggles whether to build the code templates
as part of the build process (the templates are missing critical code that
makes them uncompilable).

Templates are meant to be used as starting
code for students whereas the solution is meant for instructor use.

**IMPORTANT NOTE: The lab code output format was originally json, by default, as this was necessary for UIUC's [WebGPU](https://webgpu.com/) tool. The early goal of the Teaching Kit was to provide educators the option to use this tool, but that is no longer recommended. Text output is now the default, but you can switch back to json output by adding “-DJSON_OUTPUT” to CMAKE_CXX_FLAGS.
You need to use “t” to show the advanced mode that includes CMAKE_CXX_FLAGS.**

If you have modified the above, then you should type `g` to regenerate the Makefile and then `q` to quit out of `ccmake`.
You can then use the `make` command to build the labs.

![make](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.11.15.png)

The `make` scripts builds the executables which can be run using the command template
provided in the lab's description. Here we run the `DeviceQuery` lab.

![device-query-osx](https://s3.amazonaws.com/gpuedx/resources/screenshots/Screenshot+2015-10-23+12.12.28.png)

### Windows

The usage of CMake on windows is the same as that on linux, except for windows we will using the GUI version (one can still use the command line version however).

First, launch the CMake gui application and set your source directory (the checkout directory) and the build directory (where you want the labs to be built).


![cmake-gui1](https://s3.amazonaws.com/gpuedx/resources/screenshots/1.PNG)

Clicking configure gives you an option to select which compiler to use to compile the labs. The compiler must be installed on the system and support by the CUDA toolkit

![compiler-select](https://s3.amazonaws.com/gpuedx/resources/screenshots/2.PNG)

The CMake system then searches the system and populates the proper options in your configuration. As a user you can override these options if needed

![compiler-options](https://s3.amazonaws.com/gpuedx/resources/screenshots/3.PNG)

Clicking `Generate` button, the CMake system creates the build scripts in the previously specified build directory. Since we selected Visual Studio, a Visual Studio solution is generated.

![vs-dir](https://s3.amazonaws.com/gpuedx/resources/screenshots/4.PNG)

Opening the Visual Studio solution, you can edit and compile all the labs

![vs-view](https://s3.amazonaws.com/gpuedx/resources/screenshots/5.PNG)

The labs are built like any Visual Studio project using the build button

![vs-build](https://s3.amazonaws.com/gpuedx/resources/screenshots/6.PNG)

Once the lab is built, it can be run. Here we run the device query lab

![dev-query](https://s3.amazonaws.com/gpuedx/resources/screenshots/7.PNG)

## Using Docker

[Why use NVIDIA-Docker?](https://github.com/NVIDIA/nvidia-docker/wiki/Why%20NVIDIA%20Docker)

Included with the Teaching Kit is a [Docker](http://docker.io/) build file. This file can be used to build and launch a container which contains the Teaching Kit labs along with all the software required to run them. Using a GPU within Docker is only supported on Linux, and we recommend using [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker) to run the Docker image. To build the Docker container:

~~~
docker build . -t gputeachingkit
~~~

Once built, the `gputeachingkit` image will be listed by the `docker images` command. Launching the Docker container locally with GPU support is best accomplished using [NVIDIA-Docker](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker#running-it-locally). 

**Launching the Docker container on AWS and other GPUs in the cloud is also supported by NVIDIA-Docker and you can refer to their [wiki](https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Amazon-EC2) for details.** 

For an overview of NVIDIA-Docker, please see their [blog post](https://devblogs.nvidia.com/parallelforall/nvidia-docker-gpu-server-application-deployment-made-easy/).

By default, docker volumes are not persistent. To preserve your code across docker sessions, a volume must be shared between the host system and the container. The following command mounts `$HOME/teachingkit_src` on the host system to `/opt/teachingkit/src` in the container.

~~~
nvidia-docker run -v $HOME/teachingkit_src:/opt/teachingkit/src -it gputeachingkit 
~~~

The [Docker documentation](https://docs.docker.com/engine/tutorials/dockervolumes/) has more details on how to manage docker volumes.

## NVIDIA DLI Online Courses and Certification

The NVIDIA Accelerated Computing Teaching Kit includes access to free online DLI courses – **a value of up to $90 per person per course**

## About the NVIDIA Deep Learning Institute (DLI)
The NVIDIA Deep Learning Institute (DLI) offers hands-on training for developers, datascientists, and researchers looking to solve challenging problems with deep learning and accelerated computing. Through built-in assessments, students can earn certifications thatprove subject matter competency and can be leveraged for professional career growth.

#### Attend Instructor-led Training
In addition to online, self-paced courses, DLI offers all fundamentals and industry-specific courses as in-person workshops led by DLI-certified instructors. View upcoming workshops near you at [www.nvidia.com/dli](https://www.nvidia.com/dli).

Want to schedule training for your school? Request an onsite workshop at your university at [www.nvidia.com/requestdli](https://www.nvidia.com/requestdli).
