#ifndef GPU_RTSAMPLER_H
#define GPU_RTSAMPLER_H
#include "../common/Setting.h"
//
#include <device_launch_parameters.h>

struct Camera;
struct RTHostData;
__global__ void IPRSampler(int width, int height, int seed, int spp,
                           int Sampled, int MST, int root, float* output,
                           curandState* const rngStates, Camera* camera,
                           RTHostData host_data);

#endif  // GPU_RTSAMPLER_H