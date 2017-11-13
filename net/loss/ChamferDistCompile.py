# -*- coding: utf-8 -*-
import os;
nvcc = "/usr/local/cuda-8.0/bin/nvcc";
cxx = "g++";
cudalib = "/usr/local/cuda-8.0/lib64/";
tensorflow = "~/anaconda2/lib/python2.7/site-packages/tensorflow/include";

os.system(nvcc+" -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o ./ChamferDist.cu.o ./ChamferDist.cu -I "+tensorflow+" -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2");
os.system(cxx+" -std=c++11 ./ChamferDist.cpp ./ChamferDist.cu.o -o ./ChamferDist.so -shared -fPIC -I "+tensorflow+" -lcudart -L "+cudalib+" -O2 -D_GLIBCXX_USE_CXX11_ABI=0");

