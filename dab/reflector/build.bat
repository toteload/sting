@echo off

clang-cl -o dab_gen_reflect_data.exe -W4 -Zi -O2 -Oi -IZ:\dab -D_CRT_SECURE_NO_WARNINGS main.cpp
