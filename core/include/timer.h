// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CORE_INCLUDE_TIMER_H_
#define CORE_INCLUDE_TIMER_H_

#include <sys/time.h>
#include <numeric>
#include <map>
#include <chrono>

namespace dnnmark {

// Class StopWatch
class StopWatch {
private:
  struct timeval start_;
  struct timeval end_;
  bool started_;

public:
  StopWatch() :
    started_(false) {}

  void Start() {
    if (!started_)
      gettimeofday(&start_, NULL);
    else
      LOG(FATAL) << "The Stop Watch has already started";
    started_ = true;
  }
  void Stop() {
    if (started_) {
      gettimeofday(&end_, NULL);
      started_ = false;
    } else {
      LOG(FATAL) << "No Stop Watch has been started yet";
    }
  }
  double DiffInMs() {
    return static_cast<double>(end_.tv_sec * 1000 +
                               static_cast<double>(end_.tv_usec) / 1000) -
           static_cast<double>(start_.tv_sec * 1000 +
                               static_cast<double>(start_.tv_usec) / 1000);
  }
};

class Timer {
private:
  StopWatch watch_;
  std::vector<std::string> layer_table_;
  std::vector<long double> timing_table_;
  // std::vector<long double> ns_table_;
  std::vector<double> cpu_times;
  int num_records_;
  double total_time_;

public:  
  long double cputime_elapsed; //, gputime_elapsed;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_cpugpu, cpu_finish, gpu_finish;
  Timer()
    : watch_(), num_records_(0), total_time_(0.0) {}

  std::chrono::time_point<std::chrono::high_resolution_clock> get_time_nanosec(void) {
    return std::chrono::high_resolution_clock::now();
    // return static_cast<long double>(
    //          std::chrono::duration_cast<std::chrono::nanoseconds>(
    //            std::chrono::steady_clock::now().time_since_epoch()).count());
  }

  long double timediff_nanosec(
      std::chrono::time_point<std::chrono::high_resolution_clock> t1,
      std::chrono::time_point<std::chrono::high_resolution_clock> t0) {
    return static_cast<long double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
  }

  void Start(const std::string &layer) {
    layer_table_.push_back(layer);
    // Record start time
    start_cpugpu = get_time_nanosec();    
    watch_.Start();
  }

  void StopCPU(const std::string & layer) {
    if (!layer.compare(layer_table_.back())) {
      // Record end time
      cpu_finish = get_time_nanosec();
      cputime_elapsed = timediff_nanosec(cpu_finish, start_cpugpu);
      // LOG(INFO) << ">>> CPU time : " << cputime_elapsed << "nanos";
      cpu_times.push_back(cputime_elapsed);
    } else {
      LOG(FATAL) << "Layer to measure doesn't match";
    }
  }

  // Stop and record the current elapsed time and record it
  void Stop(const std::string &layer) {
    watch_.Stop();
    // gpu_finish = get_time_nanosec();
    if (!layer.compare(layer_table_.back())) {
      timing_table_.push_back(watch_.DiffInMs());
      // gputime_elapsed = timediff_nanosec(gpu_finish, start_cpugpu);
      // ns_table_.push_back(gputime_elapsed);
    } else {
      LOG(FATAL) << "Layer to measure doesn't match";
    }
  }

  // Reset all the recorded value to 0
  void Clear() {
    layer_table_.clear();
    timing_table_.clear();
    cpu_times.clear();
    // ns_table_.clear();
    total_time_ = 0;
  }

  // Sum up all the recorded times and store the sum to vec_
  void SumRecords() {
    int index = 0;
    for (auto const &i : layer_table_) {
      LOG(INFO) << i << ": " << timing_table_[index] << "ms";
      total_time_ += timing_table_[index];
      index++;
    }
  }

  void PrintTimingTable() {
    int index = 0;
    for (auto const &i : layer_table_) {
      printf("Operation %s: GPU time %Lfms, CPU time %fms\n",
             i.c_str(), timing_table_[index], cpu_times[index] / 1000000.);
      index++;
    }
  }

  void PrintTimingTable(const char* prefix) {
    int index = 0;
    for (auto const &i : layer_table_) {
      printf("%s Operation %s: GPU time %Lfms, CPU time %fms\n",
             prefix, i.c_str(), timing_table_[index], cpu_times[index] / 1000000.);            
      index++;
    }

  }

  double GetTotalTime() {
    return total_time_;
  }

  int GetNumRecords() {
    return num_records_;
  }

};

} // namespace dnnmark

#endif // CORE_INCLUDE_TIMER_H_
