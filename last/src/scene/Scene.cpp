#include "Scene.h"

#include "Camera.h"

HostScene::HostScene() { camera = new Camera(); }

HostScene::~HostScene() {
  if (camera) delete camera;
}

DeviceScene::DeviceScene() {}

DeviceScene::~DeviceScene() {}