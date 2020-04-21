@echo off

:: build the CUDA files
nvcc -O3 -ptx -lineinfo -IZ:\ -Xcompiler "/W4 /MD /Oi /O2 /fp:fast" -c pathtrace.cu
nvcc -O3 -ptx -lineinfo -IZ:\ -Xcompiler "/W4 /MD /Oi /O2 /fp:fast" -c wavefront.cu

python porky.py

set INCLUDES=-I. -I.\ext\SDL2-2.0.10\include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include"
set LIBS=-libpath:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64" -libpath:.\ext\SDL2-2.0.10\lib\x64 cuda.lib cudart.lib SDL2.lib SDL2main.lib opengl32.lib

if not exist extlib.obj clang-cl -nologo -W4 -MD -Oi -O2 %INCLUDES% -c extlib.cpp

clang-cl -W4 -MD -Oi -O2 -Z7 -D_CRT_SECURE_NO_WARNINGS main.cpp extlib.obj %INCLUDES% -o sting -link %LIBS% -subsystem:console
