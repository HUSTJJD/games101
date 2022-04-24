#ifndef WINDOW_H
#define WINDOW_H

#include "../common/Setting.h"

class HostScene;
class Window {
 public:
  static float last_time;
  static GLFWwindow* window;
  static HostScene* scene;
  static void Resize(const int& width, const int& height);
  static void WindowsUpdate();

  static int hight, width;
  static int dx, dy, mouse_last_x, mouse_last_y;
  static GLbyte* data;
  static float FPS;
  static void Init(const int& init_wdith, const int& init_height);
  static void Destroy();
  static void Show(const char* title);
  static void CaculateFPS();
  static void Savepic();
};

#endif