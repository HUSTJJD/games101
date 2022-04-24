#ifndef GPU_RAY_H
#define GPU_RAY_H
#include <iostream>

#include "../common/Setting.h"
#include "../common/float3Extension.cuh"
struct Ray {
  float3 origin;
  float3 direction;
  // float3 direction_inv;
  // double t;  // transportation time,
  // double t_min, t_max;

  __device__ Ray(float3 origin, float3 direction)
      : origin(origin), direction(direction) {}

  __device__ Ray() {}

  float3 PointAtParameter(float t) const { return origin + direction * t; }

  friend std::ostream &operator<<(std::ostream &os, const Ray &r) {
    os << "[origin:=" << r.origin << ", direction=" << r.direction << "]\n";
    //  << ", time=" << r.t << "]\n";
    return os;
  }
};

#endif  // GPU_RAY_H
