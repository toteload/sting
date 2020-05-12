// Building with optimizations was quite slow with stb_image so just move
// it to a separate file and just compile once and link to it. I'm not going
// to edit this anyway :)

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#include <GL/gl.h>
#include "ext/glext.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define FAST_OBJ_IMPLEMENTATION
#include "ext/fast_obj.h"
#pragma clang diagnostic pop

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb_image.h"

#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM "porky_load.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include "ext/imgui/imconfig.h"
#include "ext/imgui/imgui.h"
#include "ext/imgui/imgui_impl_sdl.h"
#include "ext/imgui/imgui_impl_opengl3.h"

#include "ext/imgui/imgui.cpp"
#include "ext/imgui/imgui_draw.cpp"
#include "ext/imgui/imgui_demo.cpp"
#include "ext/imgui/imgui_widgets.cpp"
#include "ext/imgui/imgui_impl_sdl.cpp"
#include "ext/imgui/imgui_impl_opengl3.cpp"
