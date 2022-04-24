#ifndef GPU_WINDOW_H
#define GPU_WINDOW_H

#include "../common/Setting.h"

struct GLFWwindow;

class Window {
  static float last_time;
  static GLFWwindow* window;

 public:
  static int Height, Width;
  static int dx, dy, mouse_last_x, mouse_last_y;
  static GLbyte* Data;
  static float FPS;

  static void Init(int width, int height);
  static void Show(const char* title);
  static void CaculateFPS();
  static void Savepic();
};
#endif  //