cudnndir="${HOME}/cudnn"
echo $cudnndir
if [ ! -d ${cudnndir} ]; then
  mkdir ${cudnndir}
fi
rm -rf build && ./setup.sh CUDA && cd build && make && cd ..
