#include <iostream>
#include <chrono>
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <gflags/gflags.h>
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>
#include "helper_cuda.h"
#include "common.h"
#include "dnnmark.h"
#include "usage.h"
#include "warmup.h"


using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites ver"<< version <<": Start...";
  DNNMark<TestType> dnnmark(3);
  float run_time = 0.;
  dnnmark.ParseAllConfig(FLAGS_config);

  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up before initialisation..." << FLAGS_warmup << " times";
    int gpu_id = 0;
    int dev = gpuDeviceInit(gpu_id);
    std::string message;
    if (dev == -1) {
      exit(EXIT_FAILURE);
    } else {
      LOG(INFO) << "Initialized GPU " << gpu_id << " as " << dev;
    }

    for (int i = 0; i < 2; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      int status = warmupGPU(gpu_id, 10);  // 10 iterations, default problem sze
      if (status != 0) {
        fprintf(stderr, "Error status: %d\n",status);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      LOG(INFO) << "time: " + std::to_string(diff.count()) + " s, ";
      // message = "After :"+ std::to_string(diff.count()) + " s, "
      // printGPUStateInfo(device, message);      
    }
  }

  LOG(INFO) << "Initialisation called from benchmark";
  nvtxMark("Initialisation");
  dnnmark.Initialize();

  // Forward
  LOG(INFO) << "Initialisation FWD";
  nvtxMark("Setup Workspaces");
  dnnmark.SetupWorkspaces(0);
  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up..." << FLAGS_warmup << " times";
    for (int i = 0; i < FLAGS_warmup; i++) {
      LOG(INFO) << "Warming up run " << i;
      dnnmark.Forward();
    }
  }

  nvtxRangePush("Forward");
  cudaProfilerStart();
  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Forward();
  }
  dnnmark.GetTimer()->SumRecords();
  cudaProfilerStop();
  nvtxRangePop();
  run_time += dnnmark.GetTimer()->GetTotalTime();
  dnnmark.GetTimer()->Clear();
  dnnmark.FreeWorkspaces();

  LOG(INFO) << "Forward running time(ms): " << run_time << "\n\r";
  LOG(INFO) << "Timer check: " << dnnmark.GetTimer()->GetTotalTime();


  // Backward
  LOG(INFO) << "Initialisation BWD";
  dnnmark.SetupWorkspaces(1);

  nvtxRangePush("Backward");
  cudaProfilerStart();
  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Backward();
  }
  dnnmark.GetTimer()->SumRecords();
  cudaProfilerStop();
  nvtxRangePop();
  dnnmark.FreeWorkspaces();
  LOG(INFO) << "DNNMark suites: Tear down...";
  dnnmark.TearDown();

  LOG(INFO) << "Backward running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  run_time += dnnmark.GetTimer()->GetTotalTime();


  LOG(INFO) << "Total running time(ms): " << run_time;
  printf("Total running time(ms): %f\n", run_time);

  return 0;
}
