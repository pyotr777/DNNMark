#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"
#include "simpleCUBLAS.h"
#include "cuda_profiler_api.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark(58);
  dnnmark.ParseAllConfig(FLAGS_config);
  float run_time = 0.;

  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up before initialisation..." << FLAGS_warmup;
    for (int i = 0; i < FLAGS_warmup; i++) {
      int status = call_sgemm(0, 512);
      if (status != 0) {
        fprintf(stderr, "Error status: %d\n",status);
      }
    }
  }

  dnnmark.Initialize();
  LOG(INFO) << "Initialisation FWD";
  dnnmark.SetupWorkspaces(0);

  // Warm up
  if (FLAGS_warmup) {
    for (int i = 0; i < FLAGS_warmup; i++) {
      dnnmark.Forward();
    }
  }

  // Real benchmark
  cudaProfilerStart();
  dnnmark.GetTimer()->Clear();

  // Forward
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Forward();
  }
  dnnmark.GetTimer()->SumRecords();
  cudaProfilerStop();
  run_time += dnnmark.GetTimer()->GetTotalTime();
  dnnmark.FreeWorkspaces();

  LOG(INFO) << "Forward running time(ms): " << run_time << "\n\r";

  // Backward
  LOG(INFO) << "Initialisation BWD";
  dnnmark.SetupWorkspaces(1);

  cudaProfilerStart();
  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Backward();
  }
  dnnmark.GetTimer()->SumRecords();
  cudaProfilerStop();
  dnnmark.FreeWorkspaces();
  LOG(INFO) << "DNNMark suites: Tear down...";

  dnnmark.TearDown();
  LOG(INFO) << "Backward running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  run_time += dnnmark.GetTimer()->GetTotalTime();

  LOG(INFO) << "Total running time(ms): " << run_time;
  printf("Total running time(ms): %f\n", run_time);

  return 0;
}
