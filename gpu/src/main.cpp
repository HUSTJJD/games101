#include <iostream>

#include "common/Setting.h"
#include "cuda/RayTracer.h"
#include "scene/Scene.h"
#include "window/Engine.h"
#include "window/Window.h"
//----------------------------------------------------------------
#include "cuda/DeviceManager.h"
int main(int argc, char *argv[]) {
  Window window;
  window.Init(Setting::width, Setting::height);

  PrintDeviceInfo();

  HostScene::instance()->Load("../models/cornell/CornellBox-Original.obj");
  HostScene::instance()->Build();

  Engine::Instance()->ray_tracer = new RayTracer(true);
  Engine::Instance()->ray_tracer->Init(Window::Data, Setting::width,
                                       Setting::height);
  window.Show("hello world!");

  return 0;
}