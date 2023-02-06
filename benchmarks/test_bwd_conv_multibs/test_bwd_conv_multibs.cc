#include <iostream>
#include <regex>
#include <fstream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"
#include "warmup.h"


using namespace dnnmark;
using namespace std;


char* convert2chararr(string s) {  
  char* arr = new char[s.length()+1];
  strcpy(arr, s.c_str());
  return arr;
}

vector<int> searchMBSinFile(const std::string &config_file) {
    cout << "Reading " << config_file << endl;
    ifstream file;
    string line;
    string patt = "^n=([0-9,])+";
    regex e(patt);
    smatch m;

    file.open(config_file);
    string sx;
    if (file.good()) {
      while (getline(file, line)) {
        if (regex_search(line, m, e)) {
          for (auto x:m) {
            sx = x;
            cout << "Match in file " << sx << endl;
            break;
          }
        }
      }
    }
    // Drop the right side of n=15,20,30
    string rpart = sx.substr(sx.find('=')+1, sx.length());
    cout << "right part=" << rpart << "." << endl;
    // Convert to int array arr
    int n;
    vector<int> arr;
    stringstream strs(rpart);
    while(strs.good()) {
      string substr;
      getline(strs, substr, ',');
      n = stoi(substr);
      arr.push_back(n);
      cout << "n=" << n << endl;
    }
    // # Move config_ file to tmp file
    string tmpfile = "conv_tmp.tmp";
    string command_s = "mv " + config_file + " " + tmpfile;
    cout << command_s << endl;
    const char* command = convert2chararr(command_s);
    system(command);
    return arr;
}

string readAllFile(ifstream& is) {
  string contents;
  for (char c; is.get(c); contents.push_back(c)) {}
  return contents;
}

void findReplaceInFile(const string filename, int mbs) {
  string tmpfile = "conv_tmp.tmp";
  string command_s = "sed 's/n=[0-9,]*/n=" + to_string(mbs) + "/g' " 
  + tmpfile + " > " + filename;
  const char* command = convert2chararr(command_s);
  LOG(INFO) << "Execute: " << command;
  system(command);
  // regex e(patt);
  // ifstream in(filename);
  // string contents = readAllFile(in);
  // // Replace n=1,2,3 with n=1
  // string s3 = "n=" + to_string(mbs);
  // string contents2 = regex_replace(contents, e, s3);
  // cout << "File contents: " << endl;
  // cout << contents2;
  // cout << "---- replacing with " << s3  << endl;
  // if (pos != string::npos) {
  //   file_contents.replace(pos, s1.length(), s2);
  // }
}

int main(int argc, char **argv) {
  float run_time = 0.;
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);

  cout << "DNNMark suites version " << version << ": Start..." << endl;
  DNNMark<TestType> dnnmark;
  LOG(INFO) << "DNNMark created";
  // Layer<TestType>* layer_ = dnnmark.GetLayerByID(0);
  // LOG(INFO) << "Layer0 acquired";
  // DataDim* datadim = layer_->getInputDim();
  // cout << "Layer Top dim N " << datadim->n_ << endl;

  // MBS array
  vector<int> mbs= searchMBSinFile(FLAGS_config);
  for (int i=0; i < mbs.size(); i++) {    
    cout << "Using MBS " << mbs[i] << endl;
    findReplaceInFile(FLAGS_config, mbs[i]);
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
    if (FLAGS_detailedtime) {
      dnnmark.GetTimer()->PrintTimingTable(convert2chararr("n=" + to_string(mbs[i])));
    }
    dnnmark.GetTimer()->SumRecords();
    dnnmark.TearDown();
    run_time = dnnmark.GetTimer()->GetTotalTime();
    LOG(INFO) << "Total running time(ms): " << run_time;
    printf("BWD time(ms): %f\n", run_time);
    LOG(INFO) << "DNNMark suites: Tear down for mini-batch size " << mbs[i] << "...";
  }
  return 0;
}


