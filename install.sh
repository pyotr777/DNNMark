#!/bin/bash

# Install CUDA, cuDNN and DNNMark benchmark

usage=$(cat <<USAGEBLOCK
Usage:
$(basename $0) <remote_address>

USAGEBLOCK
)

if [[ $# -lt 1 ]]; then
    echo "$usage"
    exit 1
fi

ADDRESS=$1
shift

INSTALLERS=("../ubuntu_install_cuda9cudnn7.sh" "install_DNNMark.sh")
FILES=("../CUDNN7/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb" "../CUDNN7/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb" "../get_nvidia_versions.sh")


function copy_files {
	ADDR=$1
	shift
	FILES=("$@")
	REMOTE_FILES=$(ssh $ADDR ls -1 2>/dev/null)
	echo "Files ${FILES[@]}"
	echo "Copy to $ADDR"
	for F in ${FILES[@]}; do
		base=$(basename $F)
		COPY="yes"
		for RF in ${REMOTE_FILES}; do
			if [[ "$base" == "$RF" ]]; then
				echo "$base exists"
				COPY=""
				continue
			fi
		done
		if [[ -n "$COPY" ]]; then
			echo "Copying $F"
			scp -r $F $ADDRESS: 2>/dev/null
		fi
	done
}

echo "Installing CUDA, cuDNN and DNNMark benchmark on $ADDRESS"

copy_files "$ADDRESS" "${FILES[@]}"
copy_files "$ADDRESS" "${INSTALLERS[@]}"

set -e
for INSTALLER in ${INSTALLERS[@]}; do
    echo "********************************"
    echo "Running $INSTALLER"
    echo "********************************"
	ssh $ADDRESS ./$(basename $INSTALLER)
done
echo "Installation finished. Login with ssh $ADDRESS and run ./run_cifar.sh"
