rm -rf build
mkdir build
cd build
cmake -DENABLE_CUDA=ON ..
cmake --build .

cd ..
cp build/src/Roomie* .
rm -rf build