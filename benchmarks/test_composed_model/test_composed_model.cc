#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
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
  LOG(INFO) << "DNNMark suites ver" << version << ": Start...";
  DNNMark<TestType> dnnmark(3);
  float run_time = 0.;
  dnnmark.ParseAllConfig(FLAGS_config);

  warmup(FLAGS_warmup, 0, std::string("Warming up before initialization..."));

  LOG(INFO) << "Initialisation called from benchmark";
  nvtxMark("Initialisation");
  dnnmark.Initialize();

  // Forward
  LOG(INFO) << "Initialisation FWD";
  nvtxMark("Setup Workspaces");
  dnnmark.SetupWorkspaces(0);
  warmup(FLAGS_warmup, 0, std::string("Warming up..."));


  LOG(INFO) << "Iterations " << FLAGS_iterations;
  LOG(INFO) << "Cached Iterations " << FLAGS_cachediterations;
  int slowiterations = 1;
  int fastiterations = 1;
  if (FLAGS_cachediterations) {
    fastiterations = FLAGS_iterations;
  } else {
    slowiterations = FLAGS_iterations;
  }
  nvtxRangePush("Forward");
  cudaProfilerStart();
  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < slowiterations; i++) {
    dnnmark.Forward(fastiterations);
  }
  if (FLAGS_detailedtime) {
    dnnmark.GetTimer()->PrintTimingTable();
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
  for (int i = 0; i < slowiterations; i++) {
    dnnmark.Backward(fastiterations);
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
