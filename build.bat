@echo off

:: build the CUDA files
::nvcc -O3 -Xcompiler "/MD /Oi" -c fillimage.cu

::python genglextensions.py

::clang-cl -MD -Oi -O2 main.cpp fillimage.obj -o sting -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64\cudart.lib" ext\SDL2-2.0.10\lib\x64\SDL2.lib ext\SDL2-2.0.10\lib\x64\SDL2main.lib opengl32.lib -link -subsystem:console

::clang-cl -W4 -MD -Oi -O2 -c extlib.cpp

clang-cl -W4 -MD -Oi -O2 main.cpp extlib.obj fillimage.obj -o sting -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64\cudart.lib" ext\SDL2-2.0.10\lib\x64\SDL2.lib ext\SDL2-2.0.10\lib\x64\SDL2main.lib opengl32.lib -link -subsystem:console
