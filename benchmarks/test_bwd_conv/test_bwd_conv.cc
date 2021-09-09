#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"
#include "warmup.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  float run_time = 0.;
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites version "<< version <<": Start...";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseGeneralConfig(FLAGS_config);
  dnnmark.ParseLayerConfig(FLAGS_config);
  warmup(FLAGS_warmup, 0, std::string("Warming up before initialization..."));
  LOG(INFO) << "Start initialization (dnnmark.Initialize)";
  dnnmark.Initialize();
  dnnmark.SetupWorkspaces(1);// 0 - forward, 1 - backward, 2 - forward and backward
  LOG(INFO) << "initialization done.";

  // Warmup
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
  // Real benchmark
  dnnmark.GetTimer()->Clear();
  for (int i = 0; i < slowiterations; i++) {
    dnnmark.Backward(fastiterations);
  }
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  run_time = dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "Total running time(ms): " << run_time;
  printf("BWD time(ms): %f\n", run_time);
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
