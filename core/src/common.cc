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

#include "common.h"
#include <glog/logging.h>
#include <regex>
#include <fstream>

namespace dnnmark {

//
// Internal data type. Code courtesy of Caffe
//

float DataType<float>::oneval = 1.0;
float DataType<float>::zeroval = 0.0;
const void* DataType<float>::one =
    static_cast<void *>(&DataType<float>::oneval);
const void* DataType<float>::zero =
    static_cast<void *>(&DataType<float>::zeroval);
#ifdef NVIDIA_CUDNN
double DataType<double>::oneval = 1.0;
double DataType<double>::zeroval = 0.0;
const void* DataType<double>::one =
    static_cast<void *>(&DataType<double>::oneval);
const void* DataType<double>::zero =
    static_cast<void *>(&DataType<double>::zeroval);
#endif

} // namespace dnnmark

using namespace std;

char* convert2chararr(string s) {  
  char* arr = new char[s.length()+1];
  strcpy(arr, s.c_str());
  return arr;
}

vector<int> searchMBSinFile(const std::string &config_file) {
    // cout << "Reading " << config_file << endl;
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
            // cout << "Match in file " << sx << endl;
            break;
          }
        }
      }
    }
    // Drop the right side of n=15,20,30
    string rpart = sx.substr(sx.find('=')+1, sx.length());
    // cout << "right part=" << rpart << "." << endl;
    // Convert to int array arr
    int n;
    vector<int> arr;
    stringstream strs(rpart);
    while(strs.good()) {
      string substr;
      getline(strs, substr, ',');
      n = stoi(substr);
      arr.push_back(n);
      // cout << "n=" << n << endl;
    }
    // # Move config_ file to tmp file
    string tmpfile = "conv_tmp.tmp";
    string command_s = "mv " + config_file + " " + tmpfile;
    LOG(INFO) << command_s;
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
}
