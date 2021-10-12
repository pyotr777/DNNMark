#include <iostream>
#include <chrono>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <nvml.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "warmup.h"


// multiply each element of X to each element of Y and sum
__global__
void multiply(int n, int *x, int *y, int *z) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    z[i] = 0;
    for (int j = 0; j < n; j++) {
      z[i] += x[i] * y[j];
    }
    // printf("z[%d]=%d, index=%d stride=%d\n", i, z[i], index, stride);
  }
}

// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message) {
  nvmlPstates_t pstate;
  nvmlMemory_t memory;
  unsigned int temp;
  unsigned int app_clock = 0;
  unsigned int app_clock_max = 0;
  float clock_perf;
  nvmlReturn_t nvmlRet;

  nvmlRet = nvmlDeviceGetPerformanceState(device, &pstate);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &app_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &app_clock_max);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetMemoryInfo(device, &memory);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  clock_perf = (float)app_clock / (float)app_clock_max * 100.;
  // printf("%s P%d, app clock %d/%d MHz (%3.0f%%), %d˚C, memory(free,total): %llu/%llu MB\n",
  //        message.c_str(), pstate, app_clock, app_clock_max, clock_perf, temp,
  //        memory.free / 1000000, memory.total / 1000000);
  LOG(INFO) << message << " P" << pstate << ", appclock " << clock_perf << "%, " << temp << "˚C";
}


// Main warmup function
void warmup(int FLAGS_warmup, int gpu_id, std::string message) {
  LOG(INFO) << "Warmup function v.1.02";
  if (FLAGS_warmup == 0) {
    return;
  }
  LOG(INFO) << message;
  auto start = std::chrono::high_resolution_clock::now();
  int status = warmupGPU(gpu_id);
  if (status != 0) {
    fprintf(stderr, "Error status: %d\n", status);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  LOG(INFO) << "Warming up time " + std::to_string(diff.count()*1000) + " ms";
}


// Return current GPU app clock Hz % of max
float getGPUclock(nvmlDevice_t device) {
  unsigned int app_clock = 0;
  unsigned int app_clock_max = 0;
  float clock_perf;
  nvmlReturn_t nvmlRet;

  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &app_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &app_clock_max);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }

  clock_perf = (float)app_clock / (float)app_clock_max * 100.;
  return clock_perf;
}


/* Call with device number and matrix size */
int warmupGPU(int gpu_id, bool check_results, bool debug) {
  int elements_per_thread = 4;
  float target_warmup = 97.; //% of max app clock Hz
  int maxiter = 100;
  nvmlDevice_t nvmldevice;
  nvmlReturn_t nvmlRet;
  std::string message;
  char deviceName [50];
  cudaError_t error;

  // Init NVML
  nvmlRet = nvmlInit_v2();
  if (nvmlRet != 0) {
    printf("NVML init failure. Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetHandleByIndex_v2(gpu_id, &nvmldevice);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  if (debug) {
    message = "Before:";
    printGPUStateInfo(nvmldevice, message);
  }

  // Get GPU properties (Max threads, blocks etc.)
  cudaSetDevice(gpu_id);
  cudaDeviceProp dev_prop;
  cudaGetDeviceProperties(&dev_prop, gpu_id);
  int SMs = dev_prop.multiProcessorCount;
  int SMmax = dev_prop.maxThreadsPerMultiProcessor;
  int max_block_size = dev_prop.maxThreadsPerBlock;
  // get Device name
  nvmlDeviceGetName(nvmldevice, &deviceName[0], 50);
  LOG(INFO) << "GPU " << deviceName << ", " << SMs << " SMs, " << SMmax 
            << " Max threads per SM, " << max_block_size << " max threads per block";


  // Set warmup parameters
  int block_size = max_block_size;
  unsigned int N = SMmax * SMs * elements_per_thread;
  int thread_blocks = (N + block_size - 1) / block_size;

  LOG(INFO) << "Warmup parameters: N=" << N << " elements, " << elements_per_thread 
            << " array elements per thread, "  << thread_blocks << " blocks x "
            << block_size << " threads per block, elements/thread:"
            << elements_per_thread << std::endl;

  int *x, *y, *z, *xd, *yd, *zd;
  x = (int *)malloc(N * sizeof(int));
  y = (int *)malloc(N * sizeof(int));
  z = (int *)malloc(N * sizeof(int));
  cudaMalloc(&xd, N * sizeof(int));
  cudaMalloc(&yd, N * sizeof(int));
  cudaMalloc(&zd, N * sizeof(int));


  // initialize x and y arrays on the host
  for (unsigned long i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = 2;
    z[i] = 0;
  }

  cudaMemcpy(xd, x, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(yd, y, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(zd, z, N * sizeof(int), cudaMemcpyHostToDevice);

  // Call Warmup procedure
  float curr_clock = getGPUclock(nvmldevice);
  int i = 1;
  while (curr_clock < target_warmup and i <= maxiter) {
    auto start = std::chrono::high_resolution_clock::now();
    multiply <<< thread_blocks, block_size>>>(N, xd, yd, zd);
    cudaDeviceSynchronize();
    // Wait for GPU to finish before accessing on host
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    error = cudaGetLastError();
    if (error != 0) {
      std::cout << "CUDA error? " << error << std::endl;
      exit(EXIT_FAILURE);
    }
    curr_clock = getGPUclock(nvmldevice);
    std::cout << i << "/" << maxiter << " clock " << curr_clock << "%, time "
              << diff.count() * 1e+3 << "ms" << std::endl;
    i++;
  }
  
  if (check_results) {
    // Check for errors (all values should be 3.0f)
    cudaMemcpy(z, zd, N * sizeof(int), cudaMemcpyDeviceToHost);
    int maxError = 0;
    unsigned long correct = 2 * N;
    std::cout << "Checking result..." << std::endl;
    for (unsigned long i = 0; i < fmin(N, 10000); i++) {
      maxError = fmax(maxError, fabs(z[i] - correct));
      std::cout << "\r" << i + 1 << "/" << N;
    }
    std::cout << std::endl;
    std::cout << "Max error: " << maxError << std::endl;
  }

  cudaFree(xd);
  cudaFree(yd);
  cudaFree(zd);
  free(x);
  free(y);
  free(z);

  message = "After :";
  printGPUStateInfo(nvmldevice, message);
  // Shutdown NVML
  nvmlRet = nvmlShutdown();
  if (nvmlRet != 0) {
    printf("NVML init failure. Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  return 0;
}
