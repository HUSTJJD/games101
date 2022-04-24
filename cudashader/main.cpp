#include "./render/Render.hpp"
#include "./scene/Scene.hpp"
#include <chrono>
#include <iostream>
#include <string>
int main()
{
    auto start = std::chrono::system_clock::now();
    std::cout << "Program start" << std::endl;

    // 构建场景
    Scene scene(784, 784);
    scene.buildScene("../models/");
    scene.buildBVH();

    Renderer r;
    r.Render(scene);

    std::cout
        << "Program end" << std::endl;
    auto end = std::chrono::system_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(end - start).count() << " : "
              << std::chrono::duration_cast<std::chrono::minutes>(end - start).count() << " : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
}