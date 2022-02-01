#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites version " << version << ": Start...";
  DNNMark<TestType> dnnmark(3);
  dnnmark.ParseAllConfig(FLAGS_config);
  dnnmark.Initialize();
  LOG(INFO) << "initialization done.";

  // Warmup
  if (FLAGS_warmup) {
    LOG(INFO) << "Warming up before run... " << FLAGS_warmup;
    for (int i = 0; i < FLAGS_warmup; i++) {
      dnnmark.Forward();
    }
  }

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
    dnnmark.Forward(fastiterations);
  }
  if (FLAGS_detailedtime) {
    dnnmark.GetTimer()->PrintTimingTable();
  }
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  LOG(INFO) << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  printf("FWD time(ms): %f\n", dnnmark.GetTimer()->GetTotalTime());
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
