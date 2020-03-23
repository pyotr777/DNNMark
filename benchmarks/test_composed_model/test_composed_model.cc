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
  dnnmark.ParseAllConfig(FLAGS_config);

  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up before initialisation...";
    for (int i = 0; i < 5; i++) {
      int status = call_sgemm(0, 512);
      if (status != 0) {
        fprintf(stderr, "Error status: %d\n",status);
      }
    }
  }

  // Forward
  LOG(INFO) << "Initialisation FWD";
  dnnmark.Initialize(0);

  if (FLAGS_warmup) {
    for (int i = 0; i < 5; i++) {
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
  float run_time = dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "Forward running time(ms): " << run_time;

  LOG(INFO) << "DNNMark suites: Tear down...";
  dnnmark.TearDown();

  // Backward
  dnnmark.ParseAllConfig(FLAGS_config);
  LOG(INFO) << "Initialisation BWD";
  dnnmark.Initialize(1);

  dnnmark.GetTimer()->Clear();
  // Real benchmark
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Backward();
  }
  dnnmark.GetTimer()->SumRecords();
  LOG(INFO) << "DNNMark suites: Tear down...";
  dnnmark.TearDown();

  LOG(INFO) << "Backward running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  run_time += dnnmark.GetTimer()->GetTotalTime();


  LOG(INFO) << "Total running time(ms): " << run_time;
  printf("Total running time(ms): %f\n", run_time);

  return 0;
}
