#include "float2Extension.h"
__host__ __device__ float Dot(const float2& v1, const float2& v2) {
  return v1.x * v2.x + v1.y * v2.y;
}
