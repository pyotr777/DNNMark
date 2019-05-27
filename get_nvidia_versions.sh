#!/bin/bash

# Test installed NVIDIA drivers, CUDA and cuDNN versions on Ubuntu

# Get NVIDIA driver version
version_file="/proc/driver/nvidia/version"
NV_DRV="x"
CUDA="x"
cuDNN="x"

function return_states {
    echo "NVDRV:$NV_DRV,CUDA:$CUDA,cuDNN:$cuDNN"
    exit 0
}

if [ ! -f "$version_file" ]; then
    echo "Driver not found" 1>&2
fi

# Files should exist and permissions should be all 666
if [ "$(stat -c "%a" /dev/nvidia* | grep -v 666)" ]; then
    echo "Device files /dev/nvidia* permissions should be all 666"
    stat -c "%a %n" /dev/nvidia*
fi

# Driver version
NV_DRV=$(grep -Eo "Kernel Module\s+([0-9\.]+)" /proc/driver/nvidia/version | awk '{ print $3 }')
#echo $NV_DRV


if [ "$CUDA" == "x" ];then
    NVCC_PATH=$(which nvcc)
    if [ "$NVCC_PATH" ];then
        version=$(nvcc --version | grep -oP "release \K([0-9\.\-]*)")
        CUDA=$version
        CUDA_PATH="$(dirname $NVCC_PATH)"
        CUDA_PATH=${CUDA_PATH/\/bin/}
        echo "cuda path:$CUDA_PATH" 1>&2
    fi
fi



# Get cuDNN version
CUDNN_PKG=$(dpkg --get-selections | grep -Eo "libcudnn[0-9]\s")
if [ ! "$CUDNN_PKG" ]; then
    # echo "Looking for cudnn.h"
    CUDNN_h=$(gcc -E - <<<'#include<cudnn.h>' 2>/dev/null | grep cudnn.h)
    CUDNN_h=$(echo $CUDNN_h | awk '{ print $3}')
    # Remove quotes
    CUDNN_h=$(sed -e 's/^"//' -e 's/"$//' <<<"$CUDNN_h")
    CUDNN_VERSIONS="$(cat $CUDNN_h | grep CUDNN_MAJOR -A 2)"
    CUDNN_MAJOR="$(echo "$CUDNN_VERSIONS" | grep "CUDNN_MAJOR" | head -1 | awk '{ print $3}')"
    CUDNN_MINOR="$(echo "$CUDNN_VERSIONS" | grep "CUDNN_MINOR" | head -1 | awk '{ print $3}')"
    CUDNN_PATCH="$(echo "$CUDNN_VERSIONS" | grep "CUDNN_PATCH" | head -1 | awk '{ print $3}')"
    cuDNN="${CUDNN_MAJOR}.${CUDNN_MINOR}-${CUDNN_PATCH}"
    echo "CUDNN:$cuDNN" 1>&2
else
    cuDNN=$(dpkg -s $CUDNN_PKG | grep -i "version:" | grep -Eo "[0-9\.]+-[0-9]+")
fi

return_states