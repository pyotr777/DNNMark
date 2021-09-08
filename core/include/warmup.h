#include <nvml.h>

// multiply each element of X to each element of Y and sum
__global__
void multiply(int n, int *x, int *y, int *z);

// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message);

/* Call with device number and matrix size */
int warmupGPU(int gpu_id, int iterations, unsigned int size = 100000,
              int block_size = 256, bool check_results = false);

