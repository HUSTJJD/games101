#ifndef GPU_DEVICEMANAGER_H
#define GPU_DEVICEMANAGER_H
#include "RTDeviceData.cuh"
struct Camera;
class RayTracer;
class HostScene;

void PrintDeviceInfo();
class DeviceManager {
  curandState* rng_states;
  float* devicde_float_data;
  GLbyte* devicde_byte_data;
  float* host_float_data;
  RayTracer* ray_tracer;
  dim3 grid;
  dim3 block;

  Camera* d_camera;
  RTHostData d_data;

 public:
  DeviceManager() = default;
  ~DeviceManager() = default;
  void Init(RayTracer* ray_tracer, HostScene* scene);
  void Run(HostScene* scene);
};

#endif  // GPU_DEVICEMANAGER_H