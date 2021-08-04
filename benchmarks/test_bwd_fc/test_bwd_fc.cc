#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites version "<< version <<": Start...";
  DNNMark<TestType> dnnmark;
  dnnmark.ParseGeneralConfig(FLAGS_config);
  dnnmark.ParseLayerConfig(FLAGS_config);
  dnnmark.Initialize();
  // dnnmark.Backward();
  // dnnmark.GetTimer()->SumRecords();
  // dnnmark.TearDown();
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
  for (int i = 0; i < FLAGS_iterations; i++) {
    LOG(INFO) << "Iteration " << i;
    dnnmark.Backward();
  }
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  
  printf("Total running time(ms): %f\n", dnnmark.GetTimer()->GetTotalTime());
  LOG(INFO) << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
