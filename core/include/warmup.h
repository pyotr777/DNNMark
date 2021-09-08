// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <iostream>
// #include <chrono>
// #include <string>
// #include "nvToolsExt.h"
#include <nvml.h>
// #include <gflags/gflags.h>

// /* Includes, cuda */
// #include <cuda_runtime.h>
// #include <helper_cuda.h>


// multiply each element of X to each element of Y and sum
__global__
void multiply(int n, int *x, int *y, int *z);

// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message);

/* Call with device number and matrix size */
int warmupGPU(int gpu_id, int iterations, unsigned int size = 100000);

