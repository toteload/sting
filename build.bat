@echo off

nvcc -Xcompiler "/MD" -c fillimage.cu

::clang-cl -MD -Oi main.c fillimage.obj -o sting "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudart.lib" ext\GLFW3\glfw3.lib gdi32.lib user32.lib kernel32.lib shell32.lib opengl32.lib -link -subsystem:console
clang-cl -MD -Oi main.cpp fillimage.obj -o sting -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudart.lib" ext\SDL2-2.0.10\lib\x64\SDL2.lib ext\SDL2-2.0.10\lib\x64\SDL2main.lib opengl32.lib -link -subsystem:console
