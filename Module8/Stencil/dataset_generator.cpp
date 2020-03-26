
#include "wb.h"

#define MAX_VAL 255
#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Clamp(a, start, end) Max(Min(a, end), start)
#define value(arry, i, j, k) arry[((i)*width + (j)) * depth + (k)]

static char *base_dir;

static void compute(unsigned char *out, unsigned char *in, int width,
                    int height, int depth) {

#define out(i, j, k) value(out, i, j, k)
#define in(i, j, k) value(in, i, j, k)

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        int res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
                  in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
                  6 * in(i, j, k);
        res = Clamp(res, 0, MAX_VAL);
        // printf("%d\n", res);
        out(i, j, k) = res;
      }
    }
  }
#undef out
#undef in
}

static unsigned char *generate_data(int width, int height, int depth) {
  unsigned char *data = (unsigned char *)malloc(sizeof(unsigned char) *
                                                width * height * depth);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < depth; ++k) {
        value(data, i, j, k) = rand() % MAX_VAL;
      }
    }
  }
  return data;
}

static void write_data(char *file_name, unsigned char *data, int width,
                       int height, int depth) {
  FILE *handle = fopen(file_name, "wb");
  fprintf(handle, "S6\n");
  fprintf(handle, "%d %d %d\n", width, height, depth);
  fprintf(handle, "1\n");
  fwrite(data, depth * width * sizeof(unsigned char), height, handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int width, int height, int depth) {

  const char *dir_name =
      wbDirectory_create(wbPath_join(base_dir, datasetNum));

  char *input_file_name  = wbPath_join(dir_name, "input.ppm");
  char *output_file_name = wbPath_join(dir_name, "output.ppm");

  unsigned char *input_data  = generate_data(width, height, depth);
  unsigned char *output_data = (unsigned char *)calloc(
      sizeof(unsigned char), width * height * depth);

  compute(output_data, input_data, width, height, depth);

  write_data(input_file_name, input_data, width, height, depth);
  write_data(output_file_name, output_data, width, height, depth);
}

int main() {
  base_dir = wbPath_join(wbDirectory_current(), "Stencil", "Dataset");
  create_dataset(0, 1024, 1024, 4);
  create_dataset(1, 1024, 2048, 5);
  create_dataset(2, 1023, 9, 1048);
  create_dataset(3, 1023, 1022, 8);
  create_dataset(4, 10, 1012, 1023);
  create_dataset(5, 1003, 9, 1024);
  create_dataset(6, 6, 1021, 1241);
  create_dataset(7, 9, 9, 1241);
  create_dataset(8, 1921, 19, 1241);
  return 0;
}