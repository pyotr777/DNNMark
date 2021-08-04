#!/bin/bash
# Starting Docker container and mounting ml/ folder
cd ..
docker run -ti --rm --name mlenv --gpus all --shm-size 8GB --privileged=true --cap-add=SYS_ADMIN -v $(pwd):/host pyotr777/mlenv:latest bash

