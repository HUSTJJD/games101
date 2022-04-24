#include "../common/Setting.h"
//
#include "../scene/Scene.h"
#include "../window/Window.h"
#include "DeviceManager.h"
#include "RayTracer.h"

void RayTracer::ReSetIPR() {
  if (!Setting::IPR) return;
  sampled = 0;
}

RayTracer::RayTracer(const bool GPU) : GPU(GPU) {
  if (GPU) device_manager = new DeviceManager();
}

void RayTracer::Init(GLbyte* d, int w, int h) {
  width = w;
  height = h;
  data = d;

  if (GPU) {
    device_manager->Init(this, HostScene::instance());
  }
}

void RayTracer::Render() {
  if (Done) return;
  if (GPU) {
    int targetSample = 8;
    if (sampled < targetSample)
      device_manager->Run(HostScene::instance());
    else if (sampled == targetSample) {
      Done = true;
      Window::Savepic();
    }
  }
}