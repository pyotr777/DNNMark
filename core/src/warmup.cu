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
  printf("%s P%d, app clock %d/%d MHz (%3.0f%%), %d˚C, memory(free,total): %llu/%llu MB\n",
         message.c_str(), pstate, app_clock, app_clock_max, clock_perf, temp,
         memory.free / 1000000, memory.total / 1000000);
}


/* Call with device number and matrix size */
int warmupGPU(int gpu_id, int iterations, unsigned int size, int block_size,
              bool check_results) {
  // bool debug = false;
  nvmlDevice_t nvmldevice;
  nvmlReturn_t nvmlRet;
  std::string message;
  cudaError_t error;
  unsigned int N = size;
  int thread_blocks = (N + block_size - 1) / block_size;

  std::cout << "blocks: " << thread_blocks << " x " << block_size << ", iterations: "
            << iterations << std::endl;

  int *x, *y, *z;
  // Unified memory allocation
  cudaMallocManaged(&x, N * sizeof(int));
  cudaMallocManaged(&y, N * sizeof(int));
  cudaMallocManaged(&z, N * sizeof(int));

  // initialize x and y arrays on the host
  for (unsigned long i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
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
  message = "Before start   :";
  printGPUStateInfo(nvmldevice, message);
  // Shutdown NVML
  nvmlRet = nvmlShutdown();
  if (nvmlRet != 0) {
    printf("NVML shutdown failure. Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }

  auto start = std::chrono::high_resolution_clock::now();
  // Call Warmup procedure
  for (int i = 0; i < iterations; i++) {
    multiply <<< thread_blocks, block_size>>>(N, x, y, z);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != 0) {
      std::cout << "CUDA error? " << error << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Wait for GPU to finish before accessing on host
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "N=" << N << " " << iterations << " iterations at " << diff.count() * 1e+3 << "ms" << std::endl;
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
  message = "After warmingup:";
  printGPUStateInfo(nvmldevice, message);
  // Shutdown NVML
  nvmlRet = nvmlShutdown();
  if (nvmlRet != 0) {
    printf("NVML init failure. Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }

  // Print Z array
  // std::cout << "Z:";
  // for (int i = 0; i < fmin(N, 100); i++) {
  //   std::cout <<  z[i] << " ";
  // }
  // std::cout << std::endl;

  if (check_results) {
    // Check for errors (all values should be 3.0f)
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

  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  return 0;
}

