#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <wb.h>

int main(int argc, char *argv[]) {
  wbArg_t args;
  float *__restrict__ input1;
  float *__restrict__ input2;
  float *__restrict__ output;
  int inputLength;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  input1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  input2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  output = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ Insert vector addition code here.
  wbTime_start(GPU, "Copy to GPU, compute, and copy back to host.");
  wbTime_stop(GPU, "Copy to GPU, compute, and copy back to host.");

  wbSolution(args, output, inputLength);

  // Release memory
  free(input1);
  free(input2);
  free(output);

  return 0;
}
