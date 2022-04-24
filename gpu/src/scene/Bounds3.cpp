#include "../common/float3Extension.cuh"
#include "Bounds3.h"
#include "Triangle.hpp"

Bounds3* MakeAABB(const Triangle& triangle) {
  auto aabb = new Bounds3();
  aabb->min = Minimum(triangle.v1.point, aabb->min);
  aabb->min = Minimum(triangle.v2.point, aabb->min);
  aabb->min = Minimum(triangle.v3.point, aabb->min);

  aabb->max = Maximum(triangle.v1.point, aabb->max);
  aabb->max = Maximum(triangle.v2.point, aabb->max);
  aabb->max = Maximum(triangle.v3.point, aabb->max);

  // aabb->min -= EPSILON;
  // aabb->max += EPSILON;
  return aabb;
}

Bounds3* MakeAABB(Bounds3* a, Bounds3* b) {
  auto aabb = new Bounds3();
  aabb->min = Minimum(a->min, aabb->min);
  aabb->min = Minimum(b->min, aabb->min);
  aabb->max = Maximum(a->max, aabb->max);
  aabb->max = Maximum(b->max, aabb->max);
  aabb->min -= EPSILON;
  aabb->max += EPSILON;
  return aabb;
}