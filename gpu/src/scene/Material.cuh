//
// Created by LEI XU on 5/16/19.
//

#ifndef GPU_MATERIAL_H
#define GPU_MATERIAL_H

#include <cstring>

#include "../common/Setting.h"

#define MATERIAL_PARAMTER_COUNT 6
struct Ray;
class RTDeviceData;
class Material;

class SurfaceHitRecord {
 public:
  float t;
  float3 p;
  float3 normal;
  float2 uv;
  Material* mat_ptr;

  __device__ SurfaceHitRecord() : t(99999) {}

  __device__ SurfaceHitRecord(SurfaceHitRecord* rec) {
    t = rec->t;
    p = rec->p;
    normal = rec->normal;
    mat_ptr = rec->mat_ptr;
    uv = rec->uv;
  }
};

enum MaterialType { lambertian, metal, dielectirc, light };

class Material {
 public:
  bool BackCulling = true;
  float data[MATERIAL_PARAMTER_COUNT];
  // int type;
  MaterialType Type;

  __device__ Material() {}
  __device__ Material(MaterialType t, float d[MATERIAL_PARAMTER_COUNT]) {
    Type = t;
    memcpy(data, d, MATERIAL_PARAMTER_COUNT * sizeof(float));
  }
  __device__ bool scatter(const Ray& r_in, const SurfaceHitRecord& rec,
                          float3& attenuation, Ray& scattered,
                          float3 random_in_unit_sphere,
                          const RTDeviceData& data);
  __device__ float3 emitted(float u, float v, float3& p);
};

#endif  // GPU_MATERIAL_H
