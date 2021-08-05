#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  float run_time = 0.;
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites version "<< version <<": Start...";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseGeneralConfig(FLAGS_config);
  dnnmark.ParseLayerConfig(FLAGS_config);
  dnnmark.Initialize();
  dnnmark.SetupWorkspaces(1);// 0 - forward, 1 - backward, 2 - forward and backward
  LOG(INFO) << "initialization done.";

  // Warmup
  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up before run... " << FLAGS_warmup;
    for (int i = 0; i < FLAGS_warmup; i++) {
      dnnmark.Backward();
    }
  }

  // Real benchmark
  dnnmark.GetTimer()->Clear();
  // for (int i = 0; i < FLAGS_iterations; i++) {
  LOG(INFO) << "Iterations " << FLAGS_iterations;
  dnnmark.Backward(FLAGS_iterations);
  // }
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  run_time = dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "Total running time(ms): " << run_time;
  printf("BWD time(ms): %f\n", run_time);
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
