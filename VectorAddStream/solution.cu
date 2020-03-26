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
  
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cudaStream_t stream[4];
  float *d_A[4], *d_B[4], *d_C[4];
  int i, k, Seglen = 1024;
  int Gridlen = (Seglen - 1) / 256 + 1;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc((4*Seglen+inputLength) * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  
  for (i = 0; i < 4; i++) {
    cudaStreamCreate(&stream[i]);
    wbCheck(cudaMalloc((void **)&d_A[i], (4*Seglen+inputLength) * sizeof(float)));
    wbCheck(cudaMalloc((void **)&d_B[i], (4*Seglen+inputLength) * sizeof(float)));
    wbCheck(cudaMalloc((void **)&d_C[i], (4*Seglen+inputLength) * sizeof(float)));
  }

  for (i = 0; i < inputLength; i += Seglen * 4) {
    for (k = 0; k < 4; k++) {
      cudaMemcpyAsync(d_A[k], hostInput1 + i + k * Seglen,
                      Seglen * sizeof(float), cudaMemcpyHostToDevice,
                      stream[k]);
      cudaMemcpyAsync(d_B[k], hostInput2 + i + k * Seglen,
                      Seglen * sizeof(float), cudaMemcpyHostToDevice,
                      stream[k]);
      vecAdd<<<Gridlen, 256, 0, stream[k]>>>(d_A[k], d_B[k], d_C[k],
                                             Seglen);
    }
    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    cudaStreamSynchronize(stream[2]);
    cudaStreamSynchronize(stream[3]);
    for (k = 0; k < 4; k++) {
      cudaMemcpyAsync(hostOutput + i + k * Seglen, d_C[k],
                      Seglen * sizeof(float), cudaMemcpyDeviceToHost,
                      stream[k]);
    }
  }
  cudaDeviceSynchronize();

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  
  cudaStreamDestroy(stream[0]);
  cudaStreamDestroy(stream[1]);
  cudaStreamDestroy(stream[2]);
  cudaStreamDestroy(stream[3]);

  for (k = 0; k < 4; k++) {
    cudaFree(d_A[k]);
    cudaFree(d_B[k]);
    cudaFree(d_C[k]);
  }

  return 0;
}
