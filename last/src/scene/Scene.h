#ifndef SCENE_H
#define SCENE_H

class Camera;
class HostScene {
 public:
  Camera* camera;
  HostScene();
  ~HostScene();
};

class DeviceScene {
  DeviceScene();
  ~DeviceScene();
};
#endif