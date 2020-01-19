@echo off

clang-cl -Oi main.c "ext\SDL2-2.0.10\lib\x64\SDL2.lib" "ext\SDL2-2.0.10\lib\x64\SDL2main.lib" -o sting -link -subsystem:windows
