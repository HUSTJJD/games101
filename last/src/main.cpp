

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "common/Setting.h"
#include "window/Window.h"

int main() {
  Window::Init(Setting::width, Setting::hight);
  Window::Show("Hello World!");

  Window::Destroy();
  return 0;
}
