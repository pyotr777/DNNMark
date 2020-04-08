#include <iostream>
#include "stdio.h"
#include "common.h"
#include "dnnmark.h"
#include "usage.h"
#include <gflags/gflags.h>
#include "simpleCUBLAS.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark(3);
  float run_time = 0.;
  dnnmark.ParseAllConfig(FLAGS_config);

  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up before initialisation..." << FLAGS_warmup;
    for (int i = 0; i < FLAGS_warmup; i++) {
      int status = call_sgemm(0, 512);
      if (status != 0) {
        fprintf(stderr, "Error status: %d\n",status);
      }
    }
  }

  LOG(INFO) << "Initialisation called from benchmark";
  dnnmark.Initialize();

  // Forward
  LOG(INFO) << "Initialisation FWD";
  dnnmark.SetupWorkspaces(0);
  if (FLAGS_warmup) {
    for (int i = 0; i < FLAGS_warmup; i++) {
      LOG(INFO) << "Warming up...";
      dnnmark.Forward();
    }
  }

  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Forward();
  }
  dnnmark.GetTimer()->SumRecords();
  run_time += dnnmark.GetTimer()->GetTotalTime();
  dnnmark.FreeWorkspaces();

  LOG(INFO) << "Forward running time(ms): " << run_time << "\n\r";



  // Backward
  LOG(INFO) << "Initialisation BWD";
  dnnmark.SetupWorkspaces(1);

  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Backward();
  }
  dnnmark.GetTimer()->SumRecords();
  dnnmark.FreeWorkspaces();
  LOG(INFO) << "DNNMark suites: Tear down...";
  dnnmark.TearDown();

  LOG(INFO) << "Backward running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  run_time += dnnmark.GetTimer()->GetTotalTime();


  LOG(INFO) << "Total running time(ms): " << run_time;
  printf("Total running time(ms): %f\n", run_time);

  return 0;
}
