/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <string>
#include "nvToolsExt.h"
#include <nvml.h>
#include <gflags/gflags.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}


// Print current GPU state parameters to stdout
void printGPUStateInfo(nvmlDevice_t device, std::string message) {
  nvmlPstates_t pstate;
  nvmlMemory_t memory;
  unsigned int temp;
  unsigned int app_clock;
  unsigned int app_clock_max;
  float clock_perf;
  nvmlReturn_t nvmlRet;

  nvmlRet = nvmlDeviceGetPerformanceState(device, &pstate);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &app_clock);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &app_clock_max);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  nvmlRet = nvmlDeviceGetMemoryInfo(device, &memory);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  clock_perf = (float)app_clock / (float)app_clock_max * 100.;
  printf("%s P%d, app clock %d/%d MHz (%3.0f%%), %d˚C, memory(free,total): %llu/%llu MB\n",
         message.c_str(), pstate, app_clock, app_clock_max, clock_perf, temp, 
         memory.free / 1000000, memory.total / 1000000);
}


/* Call with device number and matrix size */
static int call_sgemm(int device, int size, int iterations = 1) {
  cublasStatus_t status;
  float *h_A;
  float *h_B;
  float *h_C;
  // float *h_C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  int n2 = size * size;
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  bool debug = false;
  cublasHandle_t handle;
  nvmlDevice_t nvmldevice;
  nvmlReturn_t nvmlRet;
  std::string message;

  // Init NVML
  nvmlInit();
  LOG(INFO) << "Initialized NVML";
  nvmlRet = nvmlDeviceGetHandleByIndex_v2(device, &nvmldevice);
  if (nvmlRet != 0) {
    printf("Ret: %d\n", nvmlRet);
    exit(EXIT_FAILURE);
  }
  message = "Before iterations:";
  printGPUStateInfo(nvmldevice, message);


  /* Initialize CUBLAS */
  if (debug) {
    printf("simpleCUBLAS test running on device %d..\n", device);
  }
  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  /* Allocate host memory for the matrices */
  h_A = reinterpret_cast<float *>(malloc(n2 * sizeof(h_A[0])));

  if (h_A == 0) {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  h_B = reinterpret_cast<float *>(malloc(n2 * sizeof(h_B[0])));

  if (h_B == 0) {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return EXIT_FAILURE;
  }

  h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++) {
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
    h_C[i] = rand() / static_cast<float>(RAND_MAX);
  }

  /* Allocate device memory for the matrices */
  if (cudaMalloc(reinterpret_cast<void **>(&d_A), n2 * sizeof(d_A[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), n2 * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), n2 * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
    return EXIT_FAILURE;
  }

  for (i = 0; i < iterations; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    /* Performs operation using plain C code */
    // simple_sgemm(size, alpha, h_A, h_B, beta, h_C);
    // h_C_ref = h_C;

    /* Performs operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, d_A,
                         size, d_B, size, &beta, d_C, size);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    LOG(INFO) << std::to_string(i) + "/" + std::to_string(iterations) + " time: " +
              std::to_string(diff.count()) + " s, ";
    // message = " " + std::to_string(diff.count()) + " s, ";
    // printGPUStateInfo(nvmldevice, message);
  }
  message = "After iterations  :";
  printGPUStateInfo(nvmldevice, message);


  /* Allocate host memory for reading back the result from device memory */
  h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Read the result back */
  status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }


  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  // free(h_C_ref);

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

