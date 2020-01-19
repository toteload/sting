@echo off

clang-cl -MD -Oi main.c -o sting ext\GLFW3\glfw3.lib gdi32.lib user32.lib kernel32.lib shell32.lib opengl32.lib -link -subsystem:console
