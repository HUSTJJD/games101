#ifndef GPU_RTDEVICEDATA_H
#define GPU_RTDEVICEDATA_H

#include "../common/Setting.h"
#include "../common/float2Extension.cuh"
#include "../scene/BVH.cuh"

class Material;

struct RTHostData {
  cudaTextureObject_t Textures[1];
  Material* Materials;
  BVH* bvh;
  bool quick;
  bool ground;
};

struct RTDeviceData {
  float3 Ambient;
  int MaxDepth;
  cudaTextureObject_t* Textures;
  Material* Materials;
  BVH* bvh;
  bool ground;

  int tidx;
  float2 Pixel;

  // Ramdom
  curandState* curand_state;
  float Seed;
  unsigned long long seed;

  __device__ RTDeviceData(curandState* _curand_state, int _tidx, float seed,
                          float2 pixel)
      : tidx(_tidx), Pixel(pixel), Seed(seed), curand_state(_curand_state) {}
  __device__ float3 SampleTexture(int index, float u, float v) const;
  __device__ float GetRandom(float offset = 0) const {
    return curand_uniform(&curand_state[tidx]);
  }

  __device__ float rand() {
    float v = sin(Seed / 100.0f * Dot(Pixel, make_float2(12.9898f, 78.233f))) *
              43758.5453f;
    const float result = v - int(v);
    Seed += 1.0f;
    return result;
  }

  __device__ float drand48() {
    seed = (0x5DEECE66DL * seed + 0xB16) & 0xFFFFFFFFFFFFL;
    return static_cast<float>(static_cast<double>(seed >> 16) /
                              static_cast<double>(0x100000000L));
  }
};

#endif  // GPU_RTDEVICEDATA_H