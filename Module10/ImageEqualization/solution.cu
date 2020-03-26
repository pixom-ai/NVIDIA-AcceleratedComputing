#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32
#define PROBABILITY(x, width, height) (x) / ((width) * (height))

#define CLAMP(x, start, end) (min(max((x), (start)), (end)))

//@@ Define histogram equalization function

#define CORRECT_COLOR(cdfVal, cdfMin)                                     \
  CLAMP(255 * ((cdfVal) - (cdfMin)) / (1 - (cdfMin)), 0.0, 255.0)

//@@ insert code here

__global__ void castChar(float *input, unsigned char *output, int width,
                         int height, int channels) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = (row * width + col) * channels;
  if (row < height && col < width) {
    for (int c = 0; c < channels; ++c) {
      output[idx + c] = (unsigned char)(255 * input[idx + c]);
    }
  }
}

__global__ void convertGrayScale(unsigned char *input,
                                 unsigned char *output, int width,
                                 int height, int channels) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int o   = row * width + col;
  unsigned int i   = o * channels;

  if (row < height && col < width) {
    output[o] = (unsigned char)(0.21 * ((float)input[i]) +
                                0.71 * ((float)input[i + 1]) +
                                0.07 * ((float)input[i + 2]));
  }
}

__global__ void histogram(unsigned char *input, unsigned int *output,
                          int width, int height) {
  __shared__ unsigned int histoPrivate[HISTOGRAM_LENGTH];
  unsigned int row        = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col        = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int i          = ((row * width) + col);
  unsigned int stridex    = blockDim.x * gridDim.x;
  unsigned int stridey    = blockDim.y * gridDim.y;
  unsigned int privateIdx = threadIdx.y * blockDim.x + threadIdx.x;

  // Initialize private histogram, local to block, to 0
  if (privateIdx < HISTOGRAM_LENGTH)
    histoPrivate[privateIdx] = 0;
  __syncthreads();

  for (int y = row; y < height; y += stridey) {
    for (int x = col; x < width; x += stridex) {
      i = ((y * width) + x);
      atomicAdd(&(histoPrivate[input[i]]), 1);
    }
  }

  // Wait for all other threads in the block to finish
  __syncthreads();
  if (privateIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[privateIdx]), histoPrivate[privateIdx]);
  }
}

__global__ void cdfScan(unsigned int *input, float *output, float width,
                        float height) {
  const unsigned int dim = HISTOGRAM_LENGTH;
  __shared__ float sharedMem[dim];
  unsigned int tx    = threadIdx.x;
  unsigned int bx    = blockIdx.x;
  unsigned int index = bx * dim + tx;

  if (index < dim) {
    sharedMem[tx] = (float)PROBABILITY((float)input[index], width, height);
  } else {
    sharedMem[tx] = 0.0f;
  }

  // Reduction phase kernel code
  for (int stride = 1; stride <= dim; stride *= 2) {
    __syncthreads();
    int i = (tx + 1) * stride * 2 - 1;
    if (i < dim) {
      sharedMem[i] += sharedMem[i - stride];
    }
  }

  // Post reduction reverse phase
  for (int stride = dim / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int i = (tx + 1) * stride * 2 - 1;
    if ((i + stride) < dim) {
      sharedMem[i + stride] += sharedMem[i];
    }
  }
  __syncthreads();
  if (index < dim) {
    output[index] = sharedMem[tx];
  }
}

__global__ void cdfReduction(float *input, float *output) {
  __shared__ float sharedMemory[HISTOGRAM_LENGTH];
  unsigned int tx = threadIdx.x;
  unsigned int dx = blockDim.x;

  if (tx < HISTOGRAM_LENGTH) {
    sharedMemory[tx] = input[tx];
  } else {
    sharedMemory[tx] = 0.0f;
  }

  if ((dx + tx) < HISTOGRAM_LENGTH) {
    sharedMemory[dx + tx] = input[dx + tx];
  } else {
    sharedMemory[dx + tx] = 0.0f;
  }

  for (unsigned int stride = dx; stride > 0; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
      sharedMemory[tx] = (sharedMemory[tx + stride] < sharedMemory[tx])
                             ? sharedMemory[tx + stride]
                             : sharedMemory[tx];
    }
  }

  if (tx == 0)
    output[0] = sharedMemory[tx];
}

__global__ void applyHistogram(unsigned char *input1, float *input2,
                               float *input3, unsigned char *output,
                               int width, int height, int channels) {
  __shared__ float cdfValues[HISTOGRAM_LENGTH];
  __shared__ float cdfMin;
  unsigned int tx         = threadIdx.x;
  unsigned int ty         = threadIdx.y;
  unsigned int row        = blockIdx.y * blockDim.y + ty;
  unsigned int col        = blockIdx.x * blockDim.x + tx;
  unsigned int i          = ((row * width) + col) * channels;
  unsigned int privateIdx = threadIdx.y * blockDim.x + threadIdx.x;

  if (privateIdx < HISTOGRAM_LENGTH) {
    cdfValues[privateIdx] = input2[privateIdx];
  }

  if (tx == 0 && ty == 0) {
    cdfMin = input3[0];
  }
  __syncthreads();

  if (row < height && col < width) {
    for (int c = 0; c < channels; ++c) {
      output[i + c] =
          (unsigned char)CORRECT_COLOR(cdfValues[input1[i + c]], cdfMin);
    }
  }
}
__global__ void castFloat(unsigned char *input, float *output, int width,
                          int height, int channels) {
  unsigned int tx  = threadIdx.x;
  unsigned int ty  = threadIdx.y;
  unsigned int bx  = blockIdx.x;
  unsigned int by  = blockIdx.y;
  unsigned int row = by * BLOCK_SIZE + ty;
  unsigned int col = bx * BLOCK_SIZE + tx;
  unsigned int i   = (row * width + col) * channels;

  if (row < height && col < width) {
    for (int c = 0; c < channels; ++c) {
      output[i + c] = (float)(input[i + c] / 255.0);
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImageData;
  unsigned char *deviceCharImage;
  unsigned char *deviceGrayImage;
  unsigned int *deviceHistogram;
  float *deviceCDF;
  float *deviceCDFMin;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage         = wbImport(inputImageFile);
  imageWidth         = wbImage_getWidth(inputImage);
  imageHeight        = wbImage_getHeight(inputImage);
  imageChannels      = wbImage_getChannels(inputImage);
  outputImage        = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ GPU solution

  //@@ Allocate GPU memory
  wbCheck(cudaMalloc((void **)&deviceImageData, imageHeight * imageWidth *
                                                    imageChannels *
                                                    sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCharImage,
                     imageHeight * imageWidth * imageChannels *
                         sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrayImage,
                     imageHeight * imageWidth * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram,
                     HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(
      cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceCDFMin, 4 * sizeof(float)));

  //@@ Copy input memory, Memset
  wbCheck(cudaMemset(deviceHistogram, 0,
                     HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMemset(deviceCharImage, 0, imageHeight * imageWidth *
                                             imageChannels *
                                             sizeof(unsigned char)));
  wbCheck(cudaMemset(deviceGrayImage, 0,
                     imageHeight * imageWidth * sizeof(unsigned char)));
  cudaMemcpy(deviceImageData, hostInputImageData,
             imageHeight * imageWidth * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions
  dim3 DimGrid(((imageWidth - 1) / BLOCK_SIZE + 1),
               ((imageHeight - 1) / BLOCK_SIZE + 1), 1);
  dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  //@@ Cast image from float to unsigned char : Kernel
  wbTime_start(GPU, "Cast image from float to unsigned char");
  castChar<<<DimGrid, DimBlock>>>(deviceImageData, deviceCharImage,
                                  imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Cast image from float to unsigned char");

  //@@ Convert the image from RGB to GrayScale : Kernel
  wbTime_start(GPU, "Convert the image from RGB to GrayScale");
  convertGrayScale<<<DimGrid, DimBlock>>>(deviceCharImage, deviceGrayImage,
                                          imageWidth, imageHeight,
                                          imageChannels);
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Convert the image from RGB to GrayScale");

  //@@ Compute the histogram of grayImage : Kernel
  wbTime_start(GPU, "Compute the histogram of grayImage");
  dim3 DimGrid2(1, 1, 1);
  histogram<<<DimGrid2, DimBlock>>>(deviceGrayImage, deviceHistogram,
                                    imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Compute the histogram of grayImage");

  //@@Compute the Cumulative Distribution Function of histogram: A scan
  //operation
  //@@ Kernel
  wbTime_start(
      GPU, "Compute the cumulative distribution function of histogram");
  cdfScan<<<1, HISTOGRAM_LENGTH>>>(deviceHistogram, deviceCDF,
                                   ((float)imageWidth) * 1.0,
                                   ((float)imageHeight) * 1.0);
  cudaDeviceSynchronize();
  wbTime_stop(GPU,
              "Compute the cumulative distribution function of histogram");

  //@@ Compute the minimum value of the CDF: A reduction operation (Kernel)
  wbTime_start(GPU, "Compute the minimum value of the CDF");
  cdfReduction<<<1, HISTOGRAM_LENGTH / 2>>>(deviceCDF, deviceCDFMin);
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Compute the minimum value of the CDF");

  //@@ Apply the histogram equalization function: Kernel
  wbTime_start(GPU, "Apply the histogram equalization function");
  applyHistogram<<<DimGrid, DimBlock>>>(
      deviceCharImage, deviceCDF, deviceCDFMin, deviceCharImage,
      imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Apply the histogram equalization function");

  //@@Cast back to float - Kernel
  wbTime_start(GPU, "Cast back to float");
  castFloat<<<DimGrid, DimBlock>>>(deviceCharImage, deviceImageData,
                                   imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Cast back to float");

  //@@ Copy the GPU memory back to the CPU
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutputImageData, deviceImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceImageData);
  cudaFree(deviceCharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  cudaFree(deviceCDFMin);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, outputImage);

  wbImage_delete(inputImage);
  wbImage_delete(outputImage);
  return 0;
}
