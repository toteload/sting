@echo off
nvcc -c test.cu
clang -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64" -lcudart main.c test.obj -o summing.exe
