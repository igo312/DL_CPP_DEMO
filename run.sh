#!/bin/bash

## yolov5's structure is similar to front-detect, so use yolov5
# dlpreprocess compile
cd dlpreprocess 
mkdir build && cd build 
cmake .. && make -j4 install 
cd .. 
rm -r build
cd ..

# compile demo
mkdir build 
cd build
cmake ..
make -j4
make install 

# go to test dir 
cd ..
export LD_LIBRARY_PATH=$(pwd)/install/bin:$LD_LIBRARY_PATH
# # if you want to change some code and do debug, you can not delete build by comment out following line
rm -r build 
cd ./install/bin

# #pipeline test 
./testpipeline_frontDetect -m ../../resource/ultralytics_yolov5s.onnx -b 1 -p ../../resource/1080p.jpg -c 80

