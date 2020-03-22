// Building with optimizations was quite slow with stb_image so just move
// it to a separate file and just compile once and link to it. I'm not going
// to edit this anyway :)

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#include <GL/gl.h>
#include "glext.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"
#pragma clang diagnostic pop

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM "gl_extension_loader.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "imgui/imconfig.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_opengl3.h"

#include "imgui/imgui.cpp"
#include "imgui/imgui_draw.cpp"
#include "imgui/imgui_demo.cpp"
#include "imgui/imgui_widgets.cpp"
#include "imgui/imgui_impl_sdl.cpp"
#include "imgui/imgui_impl_opengl3.cpp"
