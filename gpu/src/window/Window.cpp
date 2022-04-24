#include "Window.h"

#include <stdio.h>
#include <stdlib.h>

#include <ctime>
#include <iostream>
#include <regex>

#include "Engine.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int Window::Height = 0;
int Window::Width = 0;
int Window::dx = 0;
int Window::dy = 0;
int Window::mouse_last_x = -1;
int Window::mouse_last_y = -1;
float Window::last_time = 0;
float Window::FPS = 0;
GLFWwindow* Window::window = nullptr;
GLbyte* Window::Data = nullptr;

unsigned long GetTickCount() {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

static void error_callback(int error, const char* description) {
  fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static void mouse_callback(GLFWwindow* window, int button, int x, int y) {
  int mods;
  if (button == 4)
    Engine::Instance()->OnMouseScroll(1);
  else if (button == 3)
    Engine::Instance()->OnMouseScroll(-1);
}

//这个是窗口变化的回调函数。。注意输入参数是一个glfw的窗口，一个宽度和高度
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  //这个是回调函数内的内容
  //这里是将视口改成变化后的的窗口大小
  //注意需要的注册该回调函数
  // glfwSetFramebufferSizeCallback(window,
  // framebuffer_size_callback);
  //两个参数是，glfw的窗口以及回调函数
  glViewport(0, 0, width, height);
}

int saveScreenshot(const char* filename) {
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);
  int x = viewport[0];
  int y = viewport[1];
  int width = viewport[2];
  int height = viewport[3];
  char* data =
      (char*)malloc((size_t)(width * height * 3));  // 3 components (R, G, B)
  if (!data) return 0;
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
  int saved = 0;
  // int saved = stbi_write_png(filename, width, height, 3, data, 0);
  free(data);
  return saved;
}

inline void GlMouseMotion(int x, int y) {
  if (Window::mouse_last_x == -1)
    Window::mouse_last_x = x;
  else {
    Window::dx = x - Window::mouse_last_x;
    Window::mouse_last_x = x;
  }
  if (Window::mouse_last_y == -1)
    Window::mouse_last_y = y;
  else {
    Window::dy = y - Window::mouse_last_y;
    Window::mouse_last_y = y;
  }
  Engine::Instance()->OnMouseMove(Window::dx, Window::dy);
}

inline void TimerProc(int id) { Engine::Instance()->Update(); }

static float globalElapseTime;
static const float TIME_INTERVAL = 5.0f;
static float delta;
static float now;
static float previous;

void Window::Show(const char* title) {
  //类似于之前的gluinit一般采用库都需要进行初始化
  glfwSetErrorCallback(error_callback);
  if (!glfwInit()) exit(EXIT_FAILURE);
  //设置主版本号
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  //后面两个参数是设置显示屏和共享的
  window = glfwCreateWindow(Width, Height, "Simple example", NULL, NULL);
  if (!window) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  // 1. 第一步：设置当前上下文
  glfwMakeContextCurrent(window);
  // glfwGetProcAddress是对gl库dlsym的封装，gladLoadGLLoader会去尝试加载所有版本的gl库
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(EXIT_FAILURE);
  }
  // 设置swap间隔
  // glfwSwapInterval(1);

  glViewport(0, 0, Setting::width, Setting::height);

  // 回调
  glfwSetKeyCallback(window, key_callback);
  glfwSetMouseButtonCallback(window, mouse_callback);
  glfwSetFramebufferSizeCallback(window,
                                 framebuffer_size_callback);  //窗口调整大小

  //初始化计时器
  globalElapseTime = TIME_INTERVAL;
  previous = glfwGetTime();
  //渲染流程
  while (!glfwWindowShouldClose(window)) {
    // processInput(window);
    now = glfwGetTime();
    delta = now - previous;
    previous = now;
    globalElapseTime -= delta;
    if (globalElapseTime <= 1.0f) {
      std::cout << "Timer triggered: " << globalElapseTime << std::endl;

      CaculateFPS();
      Engine::Instance()->Update();
      // reset timer
      globalElapseTime = TIME_INTERVAL;
    }
    // glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    glDrawPixels(Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, Data);

    // glClear(GL_COLOR_BUFFER_BIT);

    glfwSwapBuffers(window);

    glFlush();
    glfwPollEvents();
  }

  glfwTerminate();
}

void Window::Init(int width, int height) {
  const auto pixel_length = width * height * 4;
  Data = new GLbyte[pixel_length];
  Height = height;
  Width = width;
}

void Window::CaculateFPS() {
  const auto current_time = 0.001f * GetTickCount();
  ++FPS;
  if (current_time - last_time > 1.0f) {
    last_time = current_time;
    std::string title =
        "FPS - " + std::to_string(Engine::Instance()->ray_tracer->sampled);
    glfwSetWindowTitle(window, title.data());
    FPS = 0;
  }
}

void Window::Savepic() {
  time_t curr_time;
  tm* curr_tm;
  char date_string[100];
  char time_string[100];
  time(&curr_time);
  curr_tm = localtime(&curr_time);
  strftime(date_string, 50, "%B_%d_%Y", curr_tm);
  strftime(time_string, 50, "_%T", curr_tm);
  std::string path = date_string;
  path += std::regex_replace(time_string, std::regex(":"), "_");

  stbi_flip_vertically_on_write(1);
  if (stbi_write_png(("output/" + path + "_result.png").data(), Width, Height,
                     4, Data, Width * 4))
    std::cout << "Finished Save Picture " << path << std::endl;
  else
    std::cout << "Faild to Save Picture " << path << std::endl;
}
