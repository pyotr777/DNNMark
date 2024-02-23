#!/bin/bash
# Starting Docker container and mounting ml/ folder
image="pyotr777/dnnmark:latest"
mem="16GB"
if [[ $# -ge 2 ]]; then
    mem=$2
    echo "Memory: $mem"
fi
if [ -z ${1+x} ]; 
then
    docker run -ti --rm --name mlenv --gpus all --shm-size $mem --privileged=true --cap-add=SYS_ADMIN -v $(pwd):/host $image bash;
else
    host=$1
    echo "Hostname provided: $host"
    docker run -ti --rm --name mlenv -h $host --gpus all --shm-size $mem --privileged=true --cap-add=SYS_ADMIN -v $(pwd):/host $image bash;
fi

