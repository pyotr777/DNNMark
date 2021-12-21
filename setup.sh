#! /bin/bash

if [ $# -ne 1 ]
then
  echo "[Error] The setup script requires one additional parameter specifying whether CUDA or HCC is used"
  echo "Options: [CUDA, HIP]"
  exit
fi

OPTION=$1

BUILD_DIR=build
if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

if [ ${OPTION} = "CUDA" ]
then
  CUDNN_PATH=${HOME}/cudnn
  CUDNN_V=$(dpkg -l | grep cudnn | grep hi | awk '{print $2}')
  echo "CUDNN_V=$CUDNN_V"
  CUDNN_V=${CUDNN_V/libcudnn/}
  cmake -DCUDA_ENABLE=ON -DCUDNN_ROOT=${CUDNN_PATH} -DCUDNN_V=${CUDNN_V} -DCMAKE_BUILD_TYPE=Debug ..
  # cmake -DCUDA_ENABLE=ON -DCUDNN_ROOT=${CUDNN_PATH} ..
elif [ ${OPTION} = "HIP" ]
then
  MIOPEN_PATH=/opt/rocm/miopen
  ROCBLAS_PATH=/opt/rocm/rocblas
  CXX=/opt/rocm/hcc/bin/hcc cmake \
    -DHCC_ENABLE=ON \
    -DMIOPEN_ROOT=${MIOPEN_PATH} \
    -DROCBLAS_ROOT=${ROCBLAS_PATH} \
    ..
fi
