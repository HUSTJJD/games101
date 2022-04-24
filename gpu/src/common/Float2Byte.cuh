#ifndef GPU_FLOAT2BYTE_H
#define GPU_FLOAT2BYTE_H
#include "Setting.h"

__host__ __device__ float Range(float a, float Small = 0, float Big = 1);
__global__ void Float2Byte(bool quick, int width, int sampled, int spp,
                           float* in, GLbyte* out);

#endif  // GPU_FLOAT2BYTE_H