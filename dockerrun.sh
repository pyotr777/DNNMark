#!/bin/bash
# Starting Docker container and mounting ml/ folder
image="pyotr777/cnnbench:dnnmark208"
docker run -ti --rm --name mlenv --gpus all --shm-size 8GB --privileged=true --cap-add=SYS_ADMIN -v $(pwd):/host ${image} bash

