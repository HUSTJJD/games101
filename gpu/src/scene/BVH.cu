#include <cstdlib>

#include "../common/Setting.h"
#include "../common/float3Extension.cuh"
#include "BVH.cuh"
#include "Bounds3.h"
#include "Ray.cuh"
#include "Triangle.hpp"

float drand() { return rand() / (0x7fff + 1.0); }

float3 GetMin(Triangle* tri) {
  auto min = tri->v1.point;
  min = Minimum(tri->v2.point, min);
  min = Minimum(tri->v3.point, min);
  return min;
}

int box_x_compare(const void* a, const void* b) {
  return GetMin(*(Triangle**)a).x - GetMin(*(Triangle**)b).x < 0.0 ? -1 : 1;
}
int box_y_compare(const void* a, const void* b) {
  return GetMin(*(Triangle**)a).y - GetMin(*(Triangle**)b).y < 0.0 ? -1 : 1;
}
int box_z_compare(const void* a, const void* b) {
  return GetMin(*(Triangle**)a).z - GetMin(*(Triangle**)b).z < 0.0 ? -1 : 1;
}
BVH* BuildBVH(Triangle* triangle) {
  auto bvh = new BVH();
  bvh->tri = true;
  bvh->triangle = triangle;
  bvh->left = nullptr;
  bvh->right = nullptr;
  bvh->aabb = MakeAABB(*triangle);
  return bvh;
}

BVH* BuildBVH(Triangle** list, int n) {
  auto bvh = new BVH();
  bvh->tri = false;

  if (n == 1) return BuildBVH(list[0]);
  if (n == 2) {
    // printf("����\n");
    bvh->triangle = nullptr;
    bvh->left = BuildBVH(list[0]);
    // printf("left:  %p\n", bvh->left);
    bvh->right = BuildBVH(list[1]);
    // printf("right: %p\n", bvh->right);
  } else {
    const auto axis = int(3 * drand());
    if (axis == 0)
      qsort(list, n, sizeof(Triangle*), box_x_compare);
    else if (axis == 1)
      qsort(list, n, sizeof(Triangle*), box_y_compare);
    else
      qsort(list, n, sizeof(Triangle*), box_z_compare);

    bvh->triangle = nullptr;
    bvh->left = BuildBVH(list, n / 2);
    bvh->right = BuildBVH(list + n / 2, n - n / 2);
  }
  bvh->aabb = MakeAABB(bvh->left->aabb, bvh->right->aabb);
  return bvh;
}

BVH* ToDevice(BVH* host) {
  if (host == nullptr) return nullptr;

  BVH* device;
  auto bvh = new BVH();
  bvh->tri = host->tri;
  if (host->triangle == nullptr)
    bvh->triangle = nullptr;
  else {
    cudaCheck(cudaMalloc(reinterpret_cast<void**>((&bvh->triangle)),
                         sizeof(Triangle)));
    cudaCheck(cudaMemcpy(bvh->triangle, host->triangle, sizeof(Triangle),
                         cudaMemcpyHostToDevice));
  }

  cudaCheck(cudaMalloc(reinterpret_cast<void**>(&bvh->aabb), sizeof(Bounds3)));
  cudaCheck(cudaMemcpy(bvh->aabb, host->aabb, sizeof(Bounds3),
                       cudaMemcpyHostToDevice));

  bvh->left = ToDevice(host->left);
  bvh->right = ToDevice(host->right);

  cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(BVH)));
  cudaCheck(cudaMemcpy(device, bvh, sizeof(BVH), cudaMemcpyHostToDevice));
}

void Print(BVH* bvh, bool root) {
  if (root) {
    printf("BVH Root (%f,%f,%f) ~ (%f,%f,%f)\n", bvh->aabb->min.x,
           bvh->aabb->min.y, bvh->aabb->min.z, bvh->aabb->max.x,
           bvh->aabb->max.y, bvh->aabb->max.z);
    return;
  }
  if (bvh->triangle == nullptr) {
    printf("Node [isTriangle: %s , AABB: (%f,%f,%f)(%f,%f,%f), Triangle(%p)\n",
           bvh->tri ? "true" : "false", bvh->aabb->min.x, bvh->aabb->min.y,
           bvh->aabb->min.z, bvh->aabb->max.x, bvh->aabb->max.y,
           bvh->aabb->max.z, bvh->triangle);
  } else {
    printf(
        "Leaf [isTriangle: %s , AABB: (%f,%f,%f)(%f,%f,%f), "
        "Triangle(%f,%f,%f)(%f,%f,%f)(%f,%f,%f)\n",
        bvh->tri ? "true" : "false", bvh->aabb->min.x, bvh->aabb->min.y,
        bvh->aabb->min.z, bvh->aabb->max.x, bvh->aabb->max.y, bvh->aabb->max.z,
        bvh->triangle->v1.point.x, bvh->triangle->v1.point.y,
        bvh->triangle->v1.point.z, bvh->triangle->v2.point.x,
        bvh->triangle->v2.point.y, bvh->triangle->v2.point.z,
        bvh->triangle->v3.point.x, bvh->triangle->v3.point.y,
        bvh->triangle->v3.point.z);
  }
}
