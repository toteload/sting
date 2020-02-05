@echo off

:: build the CUDA files
::nvcc -O2 -Xcompiler "/MD /Z7" -c fillimage.cu
nvcc -O2 -Xcompiler "/MD /Oi" -c fillimage.cu

::python genglextensions.py

::clang-cl -MD -Oi -O2 main.cpp fillimage.obj -o sting -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64\cudart.lib" ext\SDL2-2.0.10\lib\x64\SDL2.lib ext\SDL2-2.0.10\lib\x64\SDL2main.lib opengl32.lib -link -subsystem:console
clang-cl -W4 -MD -O2 -Oi main.cpp fillimage.obj -o sting -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudart.lib" ext\SDL2-2.0.10\lib\x64\SDL2.lib ext\SDL2-2.0.10\lib\x64\SDL2main.lib opengl32.lib -link -subsystem:console
