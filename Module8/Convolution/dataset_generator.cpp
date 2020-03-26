#include "wb.h"
#include "limits.h"

static char *base_dir;

char *generateInput(int /*datasetNum*/, char *dirName,
                    wbGenerateParams_t params) {
  char *input_file_name = wbPath_join(dirName, "input0.ppm");
  wbDataset_generate(input_file_name, wbExportKind_ppm, params);
  return input_file_name;
}

char *generateMask(int /*datasetNum*/, char *dirName) {
  // Mask generation parameters
  wbRaw_GenerateParams_t raw_params;
  raw_params.rows   = 5;
  raw_params.cols   = 5;
  raw_params.minVal = 0;
  raw_params.maxVal = 1.0f / 25.0f;
  raw_params.type   = wbType_float;

  // Generation parameters are just the image generation parameters
  wbGenerateParams_t params;
  params.raw = raw_params;

  char *mask_file_name = wbPath_join(dirName, "input1.raw");
  wbDataset_generate(mask_file_name, wbExportKind_raw, params);
  return mask_file_name;
}

float clamp(float x) {
  return std::min(std::max(x, 0.0f), 1.0f);
}

void compute(wbImage_t output, wbImage_t input, float *mask, int mask_rows,
             int mask_cols) {

  const int num_channels = 3;

  float *inputData  = wbImage_getData(input);
  float *outputData = wbImage_getData(output);

  int img_width  = wbImage_getWidth(input);
  int img_height = wbImage_getHeight(input);

  assert(img_width == wbImage_getWidth(output));
  assert(img_height == wbImage_getHeight(output));
  assert(mask_rows % 2 == 1);
  assert(mask_cols % 2 == 1);

  int mask_radius_y = mask_rows / 2;
  int mask_radius_x = mask_cols / 2;

  for (int out_y = 0; out_y < img_height; ++out_y) {
    for (int out_x = 0; out_x < img_width; ++out_x) {
      for (int c = 0; c < num_channels; ++c) { // channels
        float acc = 0;
        for (int off_y = -mask_radius_y; off_y <= mask_radius_y; ++off_y) {
          for (int off_x = -mask_radius_x; off_x <= mask_radius_x;
               ++off_x) {
            int in_y   = out_y + off_y;
            int in_x   = out_x + off_x;
            int mask_y = mask_radius_y + off_y;
            int mask_x = mask_radius_x + off_x;
            if (in_y < img_height && in_y >= 0 && in_x < img_width &&
                in_x >= 0) {
              acc +=
                  inputData[(in_y * img_width + in_x) * num_channels + c] *
                  mask[mask_y * mask_cols + mask_x];
            } else {
              acc += 0.0f;
            }
          }
        }
        // fprintf(stderr, "%f %f\n", clamp(acc));
        outputData[(out_y * img_width + out_x) * num_channels + c] =
            clamp(acc);
      }
    }
  }
}

void generate(int datasetNum, int height, int width, int minVal,
              int maxVal) {
  char *dir_name = wbPath_join(base_dir, datasetNum);

  // Image generation parameters
  wbPPM_GenerateParams_t ppm_params;
  ppm_params.height   = height;
  ppm_params.width    = width;
  ppm_params.channels = 3;
  ppm_params.minVal   = minVal;
  ppm_params.maxVal   = maxVal;

  // Generation parameters are just the image generation parameters
  wbGenerateParams_t params;
  params.ppm = ppm_params;

  char *input_image_file_name =
      generateInput(datasetNum, dir_name, params);
  char *input_mask_file_name = generateMask(datasetNum, dir_name);

  // Import mask and image
  wbImage_t inputImage = wbImport(input_image_file_name);
  int mask_rows, mask_cols;
  float *mask_data =
      (float *)wbImport(input_mask_file_name, &mask_rows, &mask_cols);

  // Create output image
  wbImage_t outputImage = wbImage_new(width, height, 3);
  compute(outputImage, inputImage, mask_data, mask_rows, mask_cols);

  // Exporto output image
  char *output_file_name = wbPath_join(dir_name, "output.ppm");
  wbExport(output_file_name, outputImage);

  free(input_image_file_name);
  free(input_mask_file_name);
  free(output_file_name);
}

int main(void) {
  base_dir = wbPath_join(wbDirectory_current(), "Convolution", "Dataset");
  generate(0, 64, 64, 0, 1);
  generate(1, 128, 64, 0, 1);
  generate(2, 64, 128, 0, 1);
  generate(3, 64, 5, 0, 1);
  generate(4, 64, 3, 0, 1);
  generate(5, 228, 128, 0, 1);
  generate(6, 28, 12, 0, 1);

  return 0;
}
