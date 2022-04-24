#include "../common/Setting.h"
//----------------------------------------------------------------
#include <helper_cuda.h>

#include "../common/Float2Byte.cuh"
#include "../scene/Camera.hpp"
#include "../scene/Material.cuh"
#include "../scene/Scene.h"
#include "DeviceManager.h"
#include "RTSampler.cuh"
#include "RayTracer.h"

void PrintDeviceInfo() {
  auto device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    printf("没有支持CUDA的设备!\n");
    return;
  }
  for (auto dev = 0; dev < device_count; dev++) {
    cudaSetDevice(dev);
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, dev);
    printf("设备 %d: \"%s\"\n", dev, device_prop.name);
    char msg[256];
    snprintf(msg, sizeof(msg),
             "global memory大小:        %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
             static_cast<unsigned long long>(device_prop.totalGlobalMem));
    printf("%s", msg);
    printf(
        "SM数:                    %2d \n每SM CUDA核心数:           %3d "
        "\n总CUDA核心数:             %d \n",
        device_prop.multiProcessorCount,
        _ConvertSMVer2Cores(device_prop.major, device_prop.minor),
        _ConvertSMVer2Cores(device_prop.major, device_prop.minor) *
            device_prop.multiProcessorCount);
    printf("静态内存大小:             %zu bytes\n", device_prop.totalConstMem);
    printf("每block共享内存大小:      %zu bytes\n",
           device_prop.sharedMemPerBlock);
    printf("每block寄存器数:          %d\n", device_prop.regsPerBlock);
    printf("线程束大小:               %d\n", device_prop.warpSize);
    printf("每处理器最大线程数:       %d\n",
           device_prop.maxThreadsPerMultiProcessor);
    printf("每block最大线程数:        %d\n", device_prop.maxThreadsPerBlock);
    printf("线程块最大维度大小        (%d, %d, %d)\n",
           device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
           device_prop.maxThreadsDim[2]);
    printf("网格最大维度大小          (%d, %d, %d)\n",
           device_prop.maxGridSize[0], device_prop.maxGridSize[1],
           device_prop.maxGridSize[2]);
    printf("\n");
  }
  printf("************设备信息打印完毕************\n\n");
  const auto error = cudaGetLastError();
  if (error != 0) printf(RED "[ERROR] Init Cuda Error %d\n" RESET, error);
}

void DeviceManager::Init(RayTracer* tracer, HostScene* scene) {
  ray_tracer = tracer;
  grid = dim3(ray_tracer->width / Setting::BlockSize,
              ray_tracer->height / Setting::BlockSize);
  block = dim3(Setting::BlockSize, Setting::BlockSize);
  const size_t newHeapSize = 4608ull * 1024ull * 1024ull;
  cudaDeviceSetLimit(cudaLimitStackSize, newHeapSize);
  host_float_data = new float[ray_tracer->width * ray_tracer->height * 4];

  cudaCheck(
      cudaMalloc(reinterpret_cast<void**>(&devicde_float_data),
                 ray_tracer->width * ray_tracer->height * 4 * sizeof(float)));

  cudaCheck(
      cudaMalloc(reinterpret_cast<void**>(&devicde_byte_data),
                 ray_tracer->width * ray_tracer->height * 4 * sizeof(GLbyte)));

  cudaCheck(cudaMalloc(reinterpret_cast<void**>(&rng_states),
                       grid.x * block.x * sizeof(curandState)));

  cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera)));

  d_data = RTHostData();

  cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_data.Materials),
                       sizeof(Material) * scene->material_count));

  for (int i = 0; i < 1; i++) d_data.Textures[i] = scene->textlist[i];

  printf(BLU "[GPU]" YEL "Transferring BVH Data...");
  d_data.bvh = ToDevice(scene->bvh);
  printf(GRN "Done\n" RESET);
}

void DeviceManager::Run(
    HostScene* scene) {  //****** 复制输入内存 host->device ******

  cudaCheck(
      cudaMemcpy(devicde_float_data, host_float_data,
                 ray_tracer->width * ray_tracer->height * 4 * sizeof(float),
                 cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_camera, scene->camera, sizeof(Camera),
                       cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(d_data.Materials, scene->materials,
                       sizeof(Material) * scene->material_count,
                       cudaMemcpyHostToDevice));

  d_data.quick = ray_tracer->IPR_Quick;
  d_data.ground = scene->ground;
  auto mst = 0;
  auto spp = 0;
  if (d_data.quick || ray_tracer->IPR_reset_once) {
    mst = 3;
    spp = 1;
  } else {
    mst = 8;
    spp = Setting::SPP;
  }
  ray_tracer->sampled += spp;

  IPRSampler<<<grid, block>>>(ray_tracer->width, ray_tracer->height,
                              (rand() / (RAND_MAX + 1.0)) * 1000, spp,
                              ray_tracer->sampled, mst, 0, devicde_float_data,
                              rng_states, d_camera, d_data);
  Float2Byte<<<grid, block>>>(d_data.quick, ray_tracer->width,
                              ray_tracer->sampled, spp, devicde_float_data,
                              devicde_byte_data);

  cudaCheck(cudaDeviceSynchronize());

  //****** 复制输出内存 Device->host ******

  cudaCheck(
      cudaMemcpy(host_float_data, devicde_float_data,
                 ray_tracer->width * ray_tracer->height * 4 * sizeof(float),
                 cudaMemcpyDeviceToHost));
  cudaCheck(
      cudaMemcpy(ray_tracer->data, devicde_byte_data,
                 ray_tracer->width * ray_tracer->height * 4 * sizeof(GLbyte),
                 cudaMemcpyDeviceToHost));
}