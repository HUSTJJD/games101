#include "Engine.h"

#include <cmath>

#include "../scene/Camera.hpp"
#include "../scene/Scene.h"
#include "Window.h"
Engine* Engine::instance = nullptr;

void Engine::Update() {
  if (ray_tracer->thingsChanged) {
    ray_tracer->IPR_Quick = true;
  } else if (ray_tracer->IPR_Quick) {
    Window::dx = 0, Window::dy = 0;
    ray_tracer->IPR_Quick = false;
    ray_tracer->IPR_reset_once = true;
    Window::mouse_last_x = Window::mouse_last_y = -1;
  }
  ray_tracer->thingsChanged = false;

  if (ray_tracer->IPR_Quick) ray_tracer->ReSetIPR();
  if (ray_tracer->IPR_reset_once) {
    ray_tracer->ReSetIPR();
    ray_tracer->IPR_reset_once = false;
  }
  ray_tracer->Render();
}

void Engine::OnMouseMove(int a, int b) {
  // ray_tracer->thingsChanged = true;
  camera_w += a * 0.01;
  camera_y += b * 0.01 * camera_r;
  if (camera_y < 0.1) camera_y = 0.1;
  auto x = cos(camera_w) * camera_r;
  auto z = sin(camera_w) * camera_r;
  HostScene::instance()->camera->Update(make_float3(x, camera_y, z),
                                        HostScene::instance()->lookat);
}

void Engine::OnMouseScroll(int a) {
  camera_r += a * 0.5f;
  OnMouseMove(0, 0);
}
