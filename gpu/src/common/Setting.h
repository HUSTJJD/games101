#ifndef GPU_SETTING_H
#define GPU_SETTING_H

#include <glad/glad.h>
//---------------------
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//
#include <cuda_runtime.h>
#include <vector_types.h>

#include <cassert>

#define cudaCheck(x)                                                    \
  {                                                                     \
    cudaError_t err = (x);                                              \
    if (err != cudaSuccess) {                                           \
      printf(RED "[ERROR] %s Line %d: Cuda Error %s\n" RESET, __FILE__, \
             __LINE__, cudaGetErrorName(err));                          \
      assert(0);                                                        \
    }                                                                   \
  }

using byte = unsigned char;
#undef M_PI
#define M_PI 3.1415926535897932384626433832795
#define INF 99999
#define EPSILON 0.001
#define HIT_EPSILON 1e-4f
#define PREVIEW_PIXEL_SIZE 8

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

enum RenderMode {
  Raster_GL,
  RT_CPU,
  RT_GPU,
  RT_GPU_IPR,
};

namespace Setting {
static RenderMode render_mode = RT_GPU_IPR;
static float FOV = 75;
static bool IPR = true;
static int SPP = 1;
static int argc;
static char* argv;
static int width = 1024 / 2, height = 1024 / 2;
const int BlockSize = 16;
}  // namespace Setting

#endif  //