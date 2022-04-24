#ifndef GPU_RAYTRACER_H
#define GPU_RAYTRACER_H
#include <GLFW/glfw3.h>

class DeviceManager;
class RayTracer {
 public:
  RayTracer() = default;
  ~RayTracer() = default;

  bool GPU, Done = false;
  DeviceManager* device_manager;
  GLbyte* data;
  int sampled = 0;
  int width, height;

  bool thingsChanged;
  bool IPR_Quick = false, IPR_reset_once = false;
  void ReSetIPR();
  explicit RayTracer(bool GPU);

  void Init(GLbyte* data, int w, int h);
  void Render();
};

#endif  // GPU_RAYTRACER_H