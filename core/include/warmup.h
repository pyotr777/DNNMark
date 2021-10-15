#include <nvml.h>

// multiply each element of X to each element of Y and sum
__global__
void multiply(int n, int *x, int *y, int *z);

// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message);

// Return current GPU app clock Hz % of max
float getGPUclock(nvmlDevice_t device);

/* Call with device number and matrix size */
int warmupGPU(int gpu_id, bool check_results = false, bool debug = false);

// Main warmup function
void warmup(int FLAGS_warmup, int gpu_id, std::string message);

// Clock frequencies
struct clocks_struct {
  unsigned int gr_clock;
  unsigned int sm_clock;
  unsigned int sm_clock_max;
  unsigned int mem_clock;
  unsigned int vid_clock;
  float clock_perf;
};

// Get clock frequencies
clocks_struct getClocks(nvmlDevice_t device);

// Other GPU parameters
struct gpu_parameters_struct {
  unsigned int temp;
  unsigned int throttle;
  unsigned int power;
};

// Get other GPU parameters
gpu_parameters_struct getGPUstate(nvmlDevice_t device);