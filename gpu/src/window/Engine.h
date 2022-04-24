#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include "../common/Setting.h"
#include "../cuda/RayTracer.h"
class Engine {
  float camera_w = M_PI / 2, camera_y = 2, camera_r = 20;
  static Engine* instance;

 public:
  Engine() {}
  ~Engine(){};

  RayTracer* ray_tracer;

  void Init();
  void Update();
  void OnMouseMove(int x, int y);
  void OnMouseScroll(int a);

  static Engine* Instance() {
    if (!instance) instance = new Engine();
    return instance;
  }
};

#endif  //