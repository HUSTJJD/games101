#ifndef GPU_BOUNDS3_H
#define GPU_BOUNDS3_H

#include <cfloat>

#include "../common/Setting.h"
#include "../common/float3Extension.cuh"

struct Triangle;
struct Bounds3;

Bounds3* MakeAABB(const Triangle& triangle);
Bounds3* MakeAABB(Bounds3* a, Bounds3* b);

struct Bounds3 {
  float3 min, max;  // two points to specify the bounding box
  Bounds3() {
    min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  }
  Bounds3(const float3& p) : min(p), max(p) {}
  Bounds3(const float3& p1, const float3& p2) {
    min = make_float3(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
    max = make_float3(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
  }
};

#endif  // GPU_BOUNDS3_H
