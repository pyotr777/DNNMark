#include <nvml.h>

// multiply each element of X to each element of Y and sum
__global__
void multiply(int n, int *x, int *y, int *z);

// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message);

// Return current GPU app clock Hz % of max
float getGPUclock(nvmlDevice_t device);

/* Call with device number and matrix size */
int warmupGPU(int gpu_id, int iterations, bool check_results = false, bool debug = false);

