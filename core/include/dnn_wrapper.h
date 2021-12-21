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

#ifndef CORE_INCLUDE_DNN_WRAPPER_H_
#define CORE_INCLUDE_DNN_WRAPPER_H_

#include <iostream>
#include <string>
#include "common.h"
#include "dnn_utility.h"
#include "data_manager.h"
#include "timer.h"

namespace dnnmark {

//
// Convolution forward/backward functions
//

template <typename T>
inline void dnnmarkConvolutionForward(const Handle &handle,
                                      RunMode mode, int idx, Timer *timer,
                                      const void *alpha,
                                      const DataTensor<T> &bottom_desc,
                                      const void *x,
                                      const ConvolutionDesc<T> &conv_desc,
                                      const void *w,
                                      ConvAlgo<T> *conv_algo,
                                      void *workspace,
                                      size_t workspace_in_bytes,
                                      const void *beta,
                                      const DataTensor<T> &top_desc,
                                      void *y, int iterations) {
#ifdef NVIDIA_CUDNN
  LOG(INFO) << "Calling cudnnConvolutionForward " << iterations << " times, workspace " << workspace_in_bytes << "B, algo " << conv_algo->GetFwdAlgo();
  ProfilerStart(handle, mode, idx, timer, "ConvFwd");
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnConvolutionForward(
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 alpha,
                 bottom_desc.Get(), x,
                 conv_desc.GetFilter(), w,
                 conv_desc.GetConv(),
                 conv_algo->GetFwdAlgo(), workspace, workspace_in_bytes,
                 beta,
                 top_desc.Get(), y));
  }
  ProfilerStop(handle, mode, idx, timer, "ConvFwd");
#endif
#ifdef AMD_MIOPEN

  conv_algo->FindFwdAlgo(handle, mode, idx,
                         bottom_desc,
                         conv_desc,
                         top_desc,
                         x, w, y,
                         workspace, workspace_in_bytes);
  ProfilerStart(handle, mode, idx, timer, "ConvFwd");
  for (int i = 0; i < iterations; i++) {
    MIOPEN_CALL(miopenConvolutionForward(
                  mode == COMPOSED ?
                  handle.GetMIOpen(idx) : handle.GetMIOpen(),
                  alpha,
                  bottom_desc.Get(), x,
                  conv_desc.GetFilter(), w,
                  conv_desc.GetConv(),
                  conv_algo->GetFwdAlgo(),
                  beta,
                  top_desc.Get(), y,
                  workspace, workspace_in_bytes));
  }
  ProfilerStop(handle, mode, idx, timer, "ConvFwd");
#endif

}

template <typename T>
inline void dnnmarkConvolutionBackwardData(const Handle &handle,
    RunMode mode, int idx, Timer *timer,
    const void *alpha,
    const DataTensor<T> &top_desc,
    const void *dy,
    const ConvolutionDesc<T> &conv_desc,
    const void *w,
    ConvAlgo<T> *conv_algo,
    void *workspace,
    size_t workspace_in_bytes,
    const void *beta,
    const DataTensor<T> &bottom_desc,
    void *dx, int iterations) {
#ifdef NVIDIA_CUDNN
  LOG(INFO) << "Calling cudnnConvolutionBackwardData " << iterations << " times, workspace " << workspace_in_bytes << "B, algo " << conv_algo->GetBwdDataAlgo();
  ProfilerStart(handle, mode, idx, timer, "ConvBwdData");
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnConvolutionBackwardData(
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 alpha,
                 conv_desc.GetFilter(), w,
                 top_desc.Get(), dy,
                 conv_desc.GetConv(),
                 conv_algo->GetBwdDataAlgo(),
                 workspace, workspace_in_bytes,
                 beta,
                 bottom_desc.Get(), dx));
  }
  ProfilerStop(handle, mode, idx, timer, "ConvBwdData");
#endif
#ifdef AMD_MIOPEN
  conv_algo->FindBwdDataAlgo(handle, mode, idx,
                             bottom_desc,
                             conv_desc,
                             top_desc,
                             dy, w, dx,
                             workspace, workspace_in_bytes);
  ProfilerStart(handle, mode, idx, timer, "ConvBwdData");
  for (int i = 0; i < iterations; i++) {
    MIOPEN_CALL(miopenConvolutionBackwardData(
                  mode == COMPOSED ?
                  handle.GetMIOpen(idx) : handle.GetMIOpen(),
                  alpha,
                  top_desc.Get(), dy,
                  conv_desc.GetFilter(), w,
                  conv_desc.GetConv(),
                  conv_algo->GetBwdDataAlgo(),
                  beta,
                  bottom_desc.Get(), dx,
                  workspace, workspace_in_bytes));
  }
  ProfilerStop(handle, mode, idx, timer, "ConvBwdData");
#endif
}

template <typename T>
inline void dnnmarkConvolutionBackwardFilter(const Handle &handle,
    RunMode mode, int idx, Timer *timer,
    const void *alpha,
    const DataTensor<T> &bottom_desc,
    const void *x,
    const DataTensor<T> &top_desc,
    const void *dy,
    const ConvolutionDesc<T> &conv_desc,
    ConvAlgo<T> *conv_algo,
    void *workspace,
    size_t workspace_in_bytes,
    const void *beta,
    void *dw,
    int iterations) {
#ifdef NVIDIA_CUDNN
  // std::string conv_algo_param;
  LOG(INFO) << "Calling cudnnConvolutionBackwardFilter " << iterations << " times, workspace " << workspace_in_bytes << ", algo " << conv_algo->GetBwdFilterAlgo();
  cudnnFilterDescriptor_t filter_t = conv_desc.GetFilter();
  ProfilerStart(handle, mode, idx, timer, "ConvBwdFilter");
  // conv_algo_param = conv_algo->GetBwdFilterAlgoParameter();
  // // std::cout << "algo_param "<< conv_algo_param <<"\n";
  // if (conv_algo_param == "autoex") {
  //   conv_algo->checkAlgo4DataShape(bottom_desc.Get(),top_desc.Get(), filter_t);
  //   // ,workspace_in_bytes);
  //   conv_algo->FindBwdFilterAlgoEx(handle, mode, idx,
  //                             bottom_desc,
  //                             conv_desc,
  //                             top_desc,
  //                             x, dy, dw,
  //                             workspace, workspace_in_bytes);

  //   LOG(INFO) << "cuDNN AUTO selected conv. bwd filter alg. to " << conv_algo->GetBwdFilterAlgo();
  //   std::cout << "cuDNN AUTO selected bwd convolution filter algorithm:"<<conv_algo->GetBwdFilterAlgo()<<"\n";
  // }
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 alpha,
                 bottom_desc.Get(), x,
                 top_desc.Get(), dy,
                 conv_desc.GetConv(),
                 conv_algo->GetBwdFilterAlgo(),
                 workspace, workspace_in_bytes,
                 beta,
                 filter_t, dw));
  }
  ProfilerStop(handle, mode, idx, timer, "ConvBwdFilter");
#endif
#ifdef AMD_MIOPEN
  conv_algo->FindBwdFilterAlgo(handle, mode, idx,
                               bottom_desc,
                               conv_desc,
                               top_desc,
                               x, dy, dw,
                               workspace, workspace_in_bytes);
  ProfilerStart(handle, mode, idx, timer, "ConvBwdFilter");
  for (int i = 0; i < iterations; i++) {
    MIOPEN_CALL(miopenConvolutionBackwardWeights(
                  mode == COMPOSED ?
                  handle.GetMIOpen(idx) : handle.GetMIOpen(),
                  alpha,
                  top_desc.Get(), dy,
                  bottom_desc.Get(), x,
                  conv_desc.GetConv(),
                  conv_algo->GetBwdFilterAlgo(),
                  beta,
                  conv_desc.GetFilter(), dw,
                  workspace, workspace_in_bytes));
  }
  ProfilerStop(handle, mode, idx, timer, "ConvBwdFilter");
#endif
}

//
// Pooling forward/backward functions
//

template <typename T>
inline void dnnmarkPoolingForward(const Handle &handle,
                                  RunMode mode, int idx,
                                  const PoolingDesc<T> &pooling_desc,
                                  const void *alpha,
                                  const DataTensor<T> &x_desc,
                                  const void *x,
                                  const void *beta,
                                  const DataTensor<T> &y_desc,
                                  void * y,
                                  Data<T> *workspace,
                                  size_t workspace_in_bytes) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnPoolingForward(
               mode == COMPOSED ? handle.GetCudnn(idx) : handle.GetCudnn(),
               pooling_desc.Get(),
               alpha,
               x_desc.Get(), x,
               beta,
               y_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenPoolingForward(
                mode == COMPOSED ? handle.GetMIOpen(idx) : handle.GetMIOpen(),
                pooling_desc.Get(),
                alpha,
                x_desc.Get(), x,
                beta,
                y_desc.Get(), y,
                false,
                workspace->Get(), workspace_in_bytes));
#endif
}

template <typename T>
inline void dnnmarkPoolingBackward(const Handle &handle,
                                   RunMode mode, int idx,
                                   const PoolingDesc<T> &pooling_desc,
                                   const void *alpha,
                                   const DataTensor<T> &y_desc,
                                   const void *y,
                                   const DataTensor<T> &dy_desc,
                                   const void *dy,
                                   const DataTensor<T> &x_desc,
                                   const void *x,
                                   const void *beta,
                                   const DataTensor<T> &dx_desc,
                                   void *dx,
                                   Data<T> *workspace) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnPoolingBackward(
               mode == COMPOSED ? handle.GetCudnn(idx) : handle.GetCudnn(),
               pooling_desc.Get(),
               alpha,
               y_desc.Get(), y,
               dy_desc.Get(), dy,
               x_desc.Get(), x,
               beta,
               dx_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenPoolingBackward(
                mode == COMPOSED ? handle.GetMIOpen() : handle.GetMIOpen(),
                pooling_desc.Get(),
                alpha,
                y_desc.Get(), y,
                dy_desc.Get(), dy,
                x_desc.Get(), x,
                beta,
                dx_desc.Get(), dx,
                workspace->Get()));
#endif
}

//
// Activation forward/backward functions
//

template <typename T>
inline void dnnmarkActivationForward(const Handle &handle,
                                     RunMode mode, int idx,
                                     const ActivationDesc<T> &activation_desc,
                                     const void *alpha,
                                     const DataTensor<T> &bottom_desc,
                                     const void *x,
                                     const void *beta,
                                     const DataTensor<T> &top_desc,
                                     void *y) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnActivationForward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               activation_desc.Get(),
               alpha,
               bottom_desc.Get(), x,
               beta,
               top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationForward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                activation_desc.Get(),
                alpha,
                bottom_desc.Get(), x,
                beta,
                top_desc.Get(), y));
#endif
}

template <typename T>
inline void dnnmarkActivationBackward(const Handle &handle,
                                      RunMode mode, int idx,
                                      const ActivationDesc<T> &activation_desc,
                                      const void *alpha,
                                      const DataTensor<T> &top_desc,
                                      const void *y,
                                      const void *dy,
                                      const void *beta,
                                      const DataTensor<T> &bottom_desc,
                                      const void *x,
                                      void *dx) {
#ifdef NVIDIA_CUDNN
  cudnnHandle_t chandle = (mode == COMPOSED ? handle.GetCudnn(idx) : handle.GetCudnn());
  cudnnTensorDescriptor_t yDesc = top_desc.Get();
  cudnnTensorDescriptor_t dyDesc = top_desc.Get();
  cudnnTensorDescriptor_t xDesc = bottom_desc.Get();
  cudnnTensorDescriptor_t dxDesc = bottom_desc.Get();
  cudnnActivationDescriptor_t activationDesc = activation_desc.Get();
  CUDNN_CALL(cudnnActivationBackward(
               chandle,
               activationDesc,
               alpha,
               yDesc, y,
               dyDesc, dy,
               xDesc, x,
               beta,
               dxDesc, dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationBackward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                activation_desc.Get(),
                alpha,
                top_desc.Get(), y,
                top_desc.Get(), dy,
                bottom_desc.Get(), x,
                beta,
                bottom_desc.Get(), dx));
#endif
}

//
// LRN forward/backward functions
//

template <typename T>
inline void dnnmarkLRNForward(const Handle &handle,
                              RunMode mode, int idx,
                              const LRNDesc<T> &lrn_desc,
                              const LRNParam &lrn_param,
                              const void *alpha,
                              const DataTensor<T> &bottom_desc,
                              const void *x,
                              const void *beta,
                              const DataTensor<T> &top_desc,
                              void *y,
                              Data<T> *workspace) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnLRNCrossChannelForward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               lrn_desc.Get(),
               lrn_param.mode_,
               alpha,
               bottom_desc.Get(), x,
               beta,
               top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenLRNForward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                lrn_desc.Get(),
                alpha,
                bottom_desc.Get(), x,
                beta,
                top_desc.Get(), y,
                true, workspace->Get()));
#endif
}

template <typename T>
inline void dnnmarkLRNBackward(const Handle &handle,
                               RunMode mode, int idx,
                               const LRNDesc<T> &lrn_desc,
                               const LRNParam &lrn_param,
                               const void *alpha,
                               const DataTensor<T> &top_desc,
                               const void *y,
                               const void *dy,
                               const void *beta,
                               const DataTensor<T> &bottom_desc,
                               const void *x,
                               void *dx,
                               Data<T> *workspace) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnLRNCrossChannelBackward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               lrn_desc.Get(),
               lrn_param.mode_,
               alpha,
               top_desc.Get(), y,
               top_desc.Get(), dy,
               bottom_desc.Get(), x,
               beta,
               bottom_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenLRNBackward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                lrn_desc.Get(),
                alpha,
                top_desc.Get(), y,
                top_desc.Get(), dy,
                bottom_desc.Get(), x,
                beta,
                bottom_desc.Get(), dx,
                workspace->Get()));
#endif
}

//
// Fully Connected forward/backward functions
//

//
// Softmax forward/backward functions
//

template <typename T>
inline void dnnmarkSoftmaxForward(const Handle &handle,
                                  RunMode mode, int idx,
                                  const SoftmaxParam &softmax_param,
                                  const void *alpha,
                                  const DataTensor<T> &bottom_desc,
                                  const void *x,
                                  const void *beta,
                                  const DataTensor<T> &top_desc,
                                  void *y) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnSoftmaxForward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               softmax_param.algo_,
               softmax_param.mode_,
               alpha,
               bottom_desc.Get(), x,
               beta,
               top_desc.Get(), y));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenSoftmaxForward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                alpha,
                bottom_desc.Get(), x,
                beta,
                top_desc.Get(), y));
#endif
}

template <typename T>
inline void dnnmarkSoftmaxBackward(const Handle &handle,
                                   RunMode mode, int idx,
                                   const SoftmaxParam &softmax_param,
                                   const void *alpha,
                                   const DataTensor<T> &top_desc,
                                   const void *y,
                                   const void *dy,
                                   const void *beta,
                                   const DataTensor<T> &bottom_desc,
                                   void *dx) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnSoftmaxBackward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               softmax_param.algo_,
               softmax_param.mode_,
               alpha,
               top_desc.Get(), y,
               top_desc.Get(), dy,
               beta,
               bottom_desc.Get(), dx));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenSoftmaxBackward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                alpha,
                top_desc.Get(), y,
                top_desc.Get(), dy,
                beta,
                bottom_desc.Get(), dx));

#endif
}

//
// Batch Normalization forward/backward functions
//

template <typename T>
inline void dnnmarkBatchNormalizationForwardTraining(
  const Handle &handle,
  RunMode mode, int idx,
  const BatchNormParam &bn_param,
  void *alpha,
  void *beta,
  const DataTensor<T> &bottom_desc,
  const void *x,
  const DataTensor<T> &top_desc,
  void *y,
  const DataTensor<T> &scale_bias_mean_var_desc,
  void *bn_scale,
  void *bn_bias,
  double exp_avg_factor,
  void *result_running_mean,
  void *result_running_var,
  double epsilon,
  void *result_save_mean,
  void *result_save_var, int iterations) {
#ifdef NVIDIA_CUDNN
  LOG(INFO) << "Calling cudnnBatchNormalizationForwardTraining " << iterations << " times";
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 bn_param.mode_,
                 alpha,
                 beta,
                 bottom_desc.Get(), x,
                 top_desc.Get(), y,
                 scale_bias_mean_var_desc.Get(),
                 bn_scale, bn_bias,
                 exp_avg_factor,
                 result_running_mean, result_running_var,
                 epsilon,
                 result_save_mean, result_save_var));
  }
#endif
#ifdef AMD_MIOPEN
  LOG(INFO) << "Calling miopenBatchNormalizationForwardTraining " << iterations << " times";
  for (int i = 0; i < iterations; i++) {
    MIOPEN_CALL(miopenBatchNormalizationForwardTraining(
                  mode == COMPOSED ?
                  handle.GetMIOpen(idx) : handle.GetMIOpen(),
                  bn_param.mode_,
                  alpha,
                  beta,
                  bottom_desc.Get(), x,
                  top_desc.Get(), y,
                  scale_bias_mean_var_desc.Get(),
                  bn_scale, bn_bias,
                  exp_avg_factor,
                  result_running_mean, result_running_var,
                  epsilon,
                  result_save_mean, result_save_var));
  }
#endif
}

#ifdef NVIDIA_CUDNN
// Pytorch-style BN layer FWD pass
template <typename T>
inline void dnnmarkBatchNormalizationForwardTrainingEx(
  const Handle &handle,
  RunMode mode, int idx,
  const BatchNormParam &bn_param,
  void *alpha,
  void *beta,
  const DataTensor<T> &bottom_desc,
  const void *xData,
  const DataTensor<T> &top_desc,
  void *yData,
  const cudnnTensorDescriptor_t &zDesc,
  const void *zData,
  const DataTensor<T> &scale_bias_mean_var_desc,
  void *bn_scale_data,
  void *bn_bias_data,
  double exp_avg_factor,
  void *result_running_mean_data,
  void *result_running_var_data,
  double epsilon,
  void *save_mean,
  void *save_var,
  const cudnnActivationDescriptor_t &activation_desc,
  void *workspace,
  size_t workspace_in_bytes,
  void *reserve_space,
  size_t reserve_space_size,
  int iterations
) {
  LOG(INFO) << "Calling cudnnBatchNormalizationForwardTrainingEx " << iterations << " times";
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnBatchNormalizationForwardTrainingEx(
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 bn_param.mode_,
                 bn_param.bnOps,
                 alpha,
                 beta,
                 bottom_desc.Get(), xData,
                 zDesc, zData,
                 top_desc.Get(), yData,
                 scale_bias_mean_var_desc.Get(),
                 bn_scale_data, bn_bias_data,
                 exp_avg_factor,
                 result_running_mean_data, result_running_var_data,
                 epsilon,
                 save_mean, save_var,
                 activation_desc,
                 workspace,
                 workspace_in_bytes,
                 reserve_space,
                 reserve_space_size));
  }
}
#endif



template <typename T>
inline void dnnmarkBatchNormalizationBackward(
  const Handle &handle,
  RunMode mode, int idx,
  const BatchNormParam &bn_param,
  const void *alpha_data_diff,
  const void *beta_data_diff,
  const void *alpha_param_diff,
  const void *beta_param_diff,
  const DataTensor<T> &bottom_desc,
  const void *x,
  void *dx,
  const DataTensor<T> &top_desc,
  const void *dy,
  const DataTensor<T> &scale_bias_mean_var_desc,
  const void *bn_scale,
  void *result_bn_scale_diff,
  void *result_bn_bias_diff,
  double epsilon,
  const void *saved_mean,
  const void *saved_var,
  int iterations) {
#ifdef NVIDIA_CUDNN
  LOG(INFO) << "Calling cudnnBatchNormalizationBackward " << iterations << " times";
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnBatchNormalizationBackward(
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 bn_param.mode_,
                 alpha_data_diff,
                 beta_data_diff,
                 alpha_param_diff,
                 beta_param_diff,
                 bottom_desc.Get(), x,
                 top_desc.Get(), dy,
                 bottom_desc.Get(), dx,
                 scale_bias_mean_var_desc.Get(),
                 bn_scale,
                 result_bn_scale_diff, result_bn_bias_diff,
                 epsilon,
                 saved_mean, saved_var));
  }
#endif
#ifdef AMD_MIOPEN
  LOG(INFO) << "Calling miopenBatchNormalizationBackward " << iterations << " times";
  for (int i = 0; i < iterations; i++) {
    MIOPEN_CALL(miopenBatchNormalizationBackward(
                  mode == COMPOSED ?
                  handle.GetMIOpen(idx) : handle.GetMIOpen(),
                  bn_param.mode_,
                  alpha_data_diff,
                  beta_data_diff,
                  alpha_param_diff,
                  beta_param_diff,
                  bottom_desc.Get(), x,
                  top_desc.Get(), dy,
                  bottom_desc.Get(), dx,
                  scale_bias_mean_var_desc.Get(),
                  bn_scale,
                  result_bn_scale_diff, result_bn_bias_diff,
                  epsilon,
                  saved_mean, saved_var));
  }
#endif
}


#ifdef NVIDIA_CUDNN
template <typename T>
inline void dnnmarkBatchNormalizationBackwardEx(
  const Handle &handle,
  RunMode mode, int idx,
  const BatchNormParam &bn_param,
  const void *alpha_data_diff,
  const void *beta_data_diff,
  const void *alpha_param_diff,
  const void *beta_param_diff,
  const DataTensor<T> &bottom_desc,
  const void *xData,
  void *dxData,
  const DataTensor<T> &top_desc,
  const void *dy,
  const DataTensor<T> &scale_bias_mean_var_desc,
  const void *bn_scale_data,
  void *result_bn_scale_diff,
  void *result_bn_bias_diff,
  double epsilon,
  const void *saved_mean,
  const void *saved_var,
  int iterations) {

  // Null pointers
  const cudnnTensorDescriptor_t yDesc = nullptr;
  const void *yData = nullptr;
  const cudnnTensorDescriptor_t dzDesc = nullptr;
  void *dzData = nullptr;
  const void *bnBiasData = nullptr;
  const cudnnActivationDescriptor_t activationDesc = nullptr;
  void *workspace = nullptr;
  size_t workSpaceSizeInBytes = 0;
  void *reserveSpace = nullptr;
  size_t reserveSpaceSizeInBytes = 0;
  LOG(INFO) << "Calling cudnnBatchNormalizationBackwardEx " << iterations << " times";
  for (int i = 0; i < iterations; i++) {
    CUDNN_CALL(cudnnBatchNormalizationBackwardEx (
                 mode == COMPOSED ?
                 handle.GetCudnn(idx) : handle.GetCudnn(),
                 bn_param.mode_,
                 bn_param.bnOps,
                 alpha_data_diff,
                 beta_data_diff,
                 alpha_param_diff,
                 beta_param_diff,
                 bottom_desc.Get(),
                 xData,
                 yDesc,
                 yData,
                 top_desc.Get(),
                 dy,
                 dzDesc,
                 dzData,
                 bottom_desc.Get(),
                 dxData,
                 scale_bias_mean_var_desc.Get(),
                 bn_scale_data,
                 bnBiasData,
                 result_bn_scale_diff,
                 result_bn_bias_diff,
                 epsilon,
                 saved_mean,
                 saved_var,
                 activationDesc,
                 workspace,
                 workSpaceSizeInBytes,
                 reserveSpace,
                 reserveSpaceSizeInBytes));
  }
}
#endif

//
// Bypass layer
//

template <typename T>
inline void dnnmarkBypassForward(const Handle &handle,
                                 RunMode mode, int idx,
                                 const BypassDesc<T> &bypass_desc,
                                 const void *alpha,
                                 const DataTensor<T> &bottom_desc,
                                 const void *x,
                                 const void *beta,
                                 const DataTensor<T> &top_desc,
                                 void *y) {
#ifdef NVIDIA_CUDNN
  CUDA_CALL(cudaMemcpy(y,
                       x,
                       sizeof(T) * bypass_desc.Get().n_
                       * bypass_desc.Get().c_
                       * bypass_desc.Get().h_
                       * bypass_desc.Get().w_,
                       cudaMemcpyDeviceToDevice
                      ));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationForward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                bypass_desc.Get(),
                alpha,
                bottom_desc.Get(), x,
                beta,
                top_desc.Get(), y));
#endif
}

template <typename T>
inline void dnnmarkBypassBackward(const Handle &handle,
                                  RunMode mode, int idx,
                                  const BypassDesc<T> &bypass_desc,
                                  const void *alpha,
                                  const DataTensor<T> &top_desc,
                                  const void *y,
                                  const void *dy,
                                  const void *beta,
                                  const DataTensor<T> &bottom_desc,
                                  const void *x,
                                  void *dx) {
#ifdef NVIDIA_CUDNN
  CUDA_CALL(cudaMemcpy(dx,
                       dy,
                       sizeof(T) * bypass_desc.Get().n_
                       * bypass_desc.Get().c_
                       * bypass_desc.Get().h_
                       * bypass_desc.Get().w_,
                       cudaMemcpyDeviceToDevice
                      ));
#endif
#ifdef AMD_MIOPEN
  MIOPEN_CALL(miopenActivationBackward(
                mode == COMPOSED ?
                handle.GetMIOpen(idx) : handle.GetMIOpen(),
                bypass_desc.Get(),
                alpha,
                top_desc.Get(), y,
                top_desc.Get(), dy,
                bottom_desc.Get(), x,
                beta,
                bottom_desc.Get(), dx));
#endif
}

//
// Dropout layer
//

template <typename T>
inline void dnnmarkDropoutForward(const Handle &handle,
                                  RunMode mode, int idx,
                                  const DropoutDesc<T> &dropout_desc,
                                  const DataTensor<T> &bottom_desc,
                                  const void *x,
                                  const DataTensor<T> &top_desc,
                                  void *y,
                                  void *reserve_space, size_t reserve_space_size) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnDropoutForward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               dropout_desc.Get(),
               bottom_desc.Get(), x,
               top_desc.Get(), y,
               reserve_space,
               reserve_space_size
             ));
#endif
#ifdef AMD_MIOPEN
#endif
}

template <typename T>
inline void dnnmarkDropoutBackward(const Handle &handle,
                                   RunMode mode, int idx,
                                   const DropoutDesc<T> &dropout_desc,
                                   const DataTensor<T> &top_desc,
                                   const void *dy,
                                   const DataTensor<T> &bottom_desc,
                                   void *dx,
                                   void *reserve_space, size_t reserve_space_size) {
#ifdef NVIDIA_CUDNN
  CUDNN_CALL(cudnnDropoutBackward(
               mode == COMPOSED ?
               handle.GetCudnn(idx) : handle.GetCudnn(),
               dropout_desc.Get(),
               top_desc.Get(), dy,
               bottom_desc.Get(), dx,
               reserve_space,
               reserve_space_size
             ));
#endif
#ifdef AMD_MIOPEN
#endif
}

} // namespace dnnmark

#endif // CORE_INCLUDE_DNN_WRAPPER_H_
