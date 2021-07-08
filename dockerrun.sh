#!/bin/bash
# Starting Docker container and mounting ml/ folder
cd ..
docker run -ti --rm --name mlenv --gpus all --shm-size 8GB -v $(pwd):/host pyotr777/mlenv:cuda10.2 bash

