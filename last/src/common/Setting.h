#ifndef SETTING_H
#define SETTING_H

#include <glad/glad.h>
//
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//
#include <cuda_runtime.h>
#include <vector_types.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

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
// #define M_PI 3.1415926535897932384626433832795
#define INF 99999
#define EPSILON 0.001
#define HIT_EPSILON 1e-4f
#define PREVIEW_PIXEL_SIZE 8

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

static int width = 1024 / 2, hight = 1024 / 2;

const int BlockSize = 16;

// camera settings
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

}  // namespace Setting

// for printing
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"

#endif