#include <iostream>
#include <chrono>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <nvml.h>
#include <unistd.h>
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


// Get GPU temperatures
gpu_parameters_struct getGPUstate(nvmlDevice_t device) {
  gpu_parameters_struct gpu_parameters;
  unsigned int temperature = 0;
  unsigned int power = 0;
  unsigned long long clock_throt_reason = 0.0;
  //nvmlClocksThrottleReasonNone                      0x0000000000000000LL
  //nvmlClocksThrottleReasonGpuIdle                   0x0000000000000001LL
  //nvmlClocksThrottleReasonApplicationsClocksSetting 0x0000000000000002LL
  //nvmlClocksThrottleReasonSwPowerCap                0x0000000000000004LL
  //nvmlClocksThrottleReasonHwSlowdown                0x0000000000000008LL
  //nvmlClocksThrottleReasonSwThermalSlowdown         0x0000000000000020LL
  //nvmlClocksThrottleReasonHwThermalSlowdown         0x0000000000000040LL
  //nvmlClocksThrottleReasonHwPowerBrakeSlowdown      0x0000000000000080LL
  //nvmlClocksThrottleReasonDisplayClockSetting       0x0000000000000100LL
  nvmlReturn_t nvmlRet;
  nvmlRet = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetCurrentClocksThrottleReasons(device, &clock_throt_reason);

  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }

  nvmlRet = nvmlDeviceGetPowerUsage(device, &power);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  gpu_parameters.temp = temperature;
  gpu_parameters.clock_throt_reason = clock_throt_reason;
  gpu_parameters.power = power;
  return gpu_parameters;
}

// Get GPU clock frequencies
clocks_struct getClocks(nvmlDevice_t device) {
  clocks_struct clocks;
  unsigned int gr_clock = 0;
  unsigned int sm_clock = 0;
  unsigned int sm_clock_max = 0;
  unsigned int mem_clock = 0;
  unsigned int vid_clock = 0;
  float clock_perf = 0;
  nvmlReturn_t nvmlRet;

  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &gr_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &sm_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &sm_clock_max);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &mem_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_VIDEO, NVML_CLOCK_ID_CURRENT, &vid_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }  
  clock_perf = (float)sm_clock / (float)sm_clock_max * 100.;

  clocks.gr_clock = gr_clock;
  clocks.sm_clock = sm_clock;
  clocks.sm_clock_max = sm_clock_max;
  clocks.mem_clock = mem_clock;
  clocks.vid_clock = vid_clock;
  clocks.clock_perf = clock_perf;

  return clocks;
}


// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message) {
  nvmlPstates_t pstate;
  nvmlMemory_t memory;
  unsigned int temp;
  clocks_struct clocks;
  nvmlReturn_t nvmlRet;

  clocks = getClocks(device);
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
  nvmlRet = nvmlDeviceGetMemoryInfo(device, &memory);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  
  std::cout << message << " P" << pstate << ", smclock " << clocks.clock_perf << "%, " << temp << "˚C"
            << " CLOCKS (graph,sm,mem,vid): " << clocks.gr_clock << "," << clocks.sm_clock << ","
            << clocks.mem_clock << "," << clocks.vid_clock << std::endl;
}


// Main warmup function
void warmup(int FLAGS_warmup, int gpu_id,  std::string message) {  
  if (FLAGS_warmup == 0) {
    return;
  }
  LOG(INFO) << "Warmup function v.2.03" ;
  LOG(INFO) << message;
  auto start = std::chrono::high_resolution_clock::now();
  int status = warmupGPU(gpu_id, FLAGS_warmup);
  if (status != 0) {
    fprintf(stderr, "Error status: %d\n", status);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  LOG(INFO) << "Warming up time " + std::to_string(diff.count()*1000) + " ms";
}


/* Call with device number and matrix size */
// target_freq  -  % of max app clock Hz
int warmupGPU(int gpu_id, int target_freq, bool check_results, bool debug) {
  int elements_per_thread = 2;
  float reached_max = 0.; // Maximum observed frequency (%)
  int maxiter = 100; // Maximum warmup iterations
  const int decrease_count_start = 10; // allow this many times of Hz not increasing before stopping warmup
  int decrease_count = decrease_count_start;
  nvmlDevice_t nvmldevice;
  nvmlReturn_t nvmlRet;
  clocks_struct clocks;
  gpu_parameters_struct gpu_parameters;
  std::string message;
  char deviceName [50];
  cudaError_t error;
  bool need_warmup = true;

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
  message = "Before:";
  printGPUStateInfo(nvmldevice, message);

  gpu_parameters = getGPUstate(nvmldevice);
  while (gpu_parameters.clock_throt_reason > 0) {
      LOG(INFO) << "GPU throttle:" << gpu_parameters.clock_throt_reason;
      need_warmup = false; // No need to warmup
      sleep(3);
      gpu_parameters = getGPUstate(nvmldevice);
  }  

  if (need_warmup) {
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
              << " Max threads per SM, " << max_block_size << " max threads per block" << std::endl;
  
  
    // Set warmup parameters
    int block_size = max_block_size;
    unsigned int N = SMmax * SMs * elements_per_thread;
    int thread_blocks = (N + block_size - 1) / block_size;
  
    LOG(INFO) << "Warmup parameters: N=" << N << " elements, " << elements_per_thread 
              << " array elements per thread, "  << thread_blocks << " blocks x "
              << block_size << " threads per block"
              << std::endl;
  
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
    clocks = getClocks(nvmldevice);
    // gpu_parameters = getGPUstate(nvmldevice);
    int i = 1;
    auto start = std::chrono::high_resolution_clock::now();
    // Repeat warming up until target Hz reached, max number of warmup iterations reached, Hz not increasing or throttle is ON
    while (clocks.clock_perf < target_freq and i <= maxiter and decrease_count > 0 and gpu_parameters.clock_throt_reason < 2) {
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
      // curr_clock = getGPUclock(nvmldevice);
      clocks = getClocks(nvmldevice);
      gpu_parameters = getGPUstate(nvmldevice);
      LOG(INFO) << i << "/" << maxiter << " clock " << clocks.clock_perf << "%, time "
                << diff.count() * 1e+3 << "ms"
                << " CLOCKS (graph,sm,mem,vid): " << clocks.gr_clock << "," << clocks.sm_clock << ","
                << clocks.mem_clock << "," << clocks.vid_clock 
                << ", temp: " << gpu_parameters.temp 
                << "˚C, pwr: " << gpu_parameters.power/1000.
                << "W, throttle: " << gpu_parameters.clock_throt_reason;
  
      // Compare frequency with previous iteration
      if (clocks.clock_perf <= reached_max) {
        decrease_count--;
        if (decrease_count <=0) {
          LOG(INFO) << "SM frequency is not increasing. Stopping warmup." << std::endl;
        }
      } else {
        decrease_count = decrease_count_start;
        reached_max = clocks.clock_perf;
      }
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
  }
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
