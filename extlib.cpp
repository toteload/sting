// Building with optimizations was quite slow with stb_image so just move
// it to a separate file and just compile once and link to it. I'm not going
// to edit this anyway :)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"
#pragma clang diagnostic pop

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
