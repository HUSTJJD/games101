#include "Window.h"

#include "../scene/Camera.h"
#include "../scene/Scene.h"
// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    Window::scene->camera->ProcessKeyboard(FORWARD, Window::deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    Window::scene->camera->ProcessKeyboard(BACKWARD, Window::deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    Window::scene->camera->ProcessKeyboard(LEFT, Window::deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    Window::scene->camera->ProcessKeyboard(RIGHT, Window::deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
  float xpos = static_cast<float>(xposIn);
  float ypos = static_cast<float>(yposIn);

  if (firstMouse) {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }

  float xoffset = xpos - lastX;
  float yoffset =
      lastY - ypos;  // reversed since y-coordinates go from bottom to top

  lastX = xpos;
  lastY = ypos;

  scene->camera->ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  scene->camera->ProcessMouseScroll(static_cast<float>(yoffset));
}

void Window::Init(const int& x, const int& y) {
  auto pixel_length = x * y * 4;
  data = new GLbyte[pixel_length];
  width = x;
  hight = y;
}

void Window::Destroy() {
  if (window) glfwDestroyWindow(window);
  if (data) delete[] data;
  if (scene) delete scene;
}

inline void Window::Resize(const int& x, const int& y) {
  width = x;
  hight = y;
  glfwReshapeWindow(width, hight);
}

inline void Window::WindowsUpdate() {
  // glClear(GL_COLOR_BUFFER_BIT);
  CaculateFPS();
  glDrawPixels(width, hight, GL_RGBA, GL_UNSIGNED_BYTE, data);
  glfwSwapBuffers(Window::window);
  glFlush();
}

unsigned long GetTickCount() {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

void Window::Show(const char* title) {
  //类似于之前的gluinit一般采用库都需要进行初始化
  // glfwSetErrorCallback(error_callback);
  if (!glfwInit()) exit(EXIT_FAILURE);
  //设置主版本号
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  //后面两个参数是设置显示屏和共享的`
  window = glfwCreateWindow(width, hight, title, NULL, NULL);
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

  glViewport(0, 0, Setting::width, Setting::hight);

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
      // Engine::Instance()->Update();
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

inline void TimerProc(int id) {
  // Engine::Instance()->Update();
  glfwPostRedisplay();
  glfwTimerFunc(1, TimerProc, 1);
}

void Window::CaculateFPS() {
  const auto current_time = 0.001f * GetTickCount();
  ++FPS;
  if (current_time - last_time > 1.0f) {
    last_time = current_time;
    std::string title =
        "ALightGPU - " + std::to_string(Engine::Instance()->RayTracer->sampled);
    // char title[35];
    // printf(title, sizeof(title), "ALightGPU");
    glfwSetWindowTitle(window, title.data());
    FPS = 0;
  }
}