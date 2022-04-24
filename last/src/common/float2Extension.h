#ifndef GPU_FLOAT2EXTENSION_H
#define GPU_FLOAT2EXTENSION_H

#include "Setting.h"

__host__ __device__ float Dot(const float2& v1, const float2& v2);
#endif  // GPU_FLOAT2EXTENSION_H