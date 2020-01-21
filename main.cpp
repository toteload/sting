#include "ext/SDL2-2.0.10/include/SDL.h"
#include <stdio.h>
#include <stdint.h>
#include <Windows.h>
#include <GL/gl.h>
#include "glext.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "vecmath.h"

typedef uint32_t u32;

const u32 SCREEN_WIDTH  = 960;
const u32 SCREEN_HEIGHT = 540;

PFNGLCREATERENDERBUFFERSPROC glCreateRenderbuffers;
PFNGLCREATEFRAMEBUFFERSPROC glCreateFramebuffers;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC glNamedFramebufferRenderbuffer;
PFNGLNAMEDRENDERBUFFERSTORAGEPROC glNamedRenderbufferStorage;
PFNGLBLITNAMEDFRAMEBUFFERPROC glBlitNamedFramebuffer;
PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;

struct Keymap {
    u32 w : 1;
    u32 s : 1;
    u32 a : 1;
    u32 d : 1;
};

void draw_test_image(cudaArray_const_t, vec4*, vec3, uint32_t, uint32_t);

int main(int argc, char** args) {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);

    // NOTE: when these are set it cannot find a suitable opengl context
    //SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 32);
    //SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 32);
    //SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 32);
    //SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 32);

    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_Window* window = SDL_CreateWindow("sting",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          SCREEN_WIDTH,
                                          SCREEN_HEIGHT,
                                          SDL_WINDOW_OPENGL);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);

    // disable the vsync for max fps
    SDL_GL_SetSwapInterval(0);

    glCreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glCreateRenderbuffers");
    glCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glCreateFramebuffers");
    glNamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC)SDL_GL_GetProcAddress("glNamedFramebufferRenderbuffer");
    glNamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC)SDL_GL_GetProcAddress("glNamedRenderbufferStorage");
    glBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC)SDL_GL_GetProcAddress("glBlitNamedFramebuffer");
    glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteRenderbuffers");
    glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteFramebuffers");

    if (!glCreateRenderbuffers) { return 1; }
    if (!glCreateFramebuffers) { return 1; }
    if (!glNamedFramebufferRenderbuffer) { return 1; }
    if (!glNamedRenderbufferStorage) { return 1; }
    if (!glBlitNamedFramebuffer) { return 1; }
    if (!glDeleteRenderbuffers) { return 1; }
    if (!glDeleteFramebuffers) { return 1; }

    GLuint framebuffer, renderbuffer;
    cudaArray* screen_array;
    cudaGraphicsResource* graphics_resource;

    glCreateRenderbuffers(1, &renderbuffer);
    glCreateFramebuffers(1, &framebuffer);
    glNamedFramebufferRenderbuffer(framebuffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
    glNamedRenderbufferStorage(renderbuffer, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT);

    cudaGraphicsGLRegisterImage(&graphics_resource,
                                renderbuffer, GL_RENDERBUFFER, 
                                cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &graphics_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&screen_array, graphics_resource, 0, 0);
    cudaGraphicsUnmapResources(1, &graphics_resource, 0);

    // Create accumulator buffer
    vec4* screen_buffer;
    cudaMalloc(&screen_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4));

    vec3 camera_pos = { 0.0f, 0.0f, 0.0f };

    Keymap keymap = { 0 };

    int running = 1;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT: { running = 0; } break;
            case SDL_KEYDOWN: {
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE: { running = 0; } break;
                case SDLK_w: { keymap.w = 1; } break;
                case SDLK_s: { keymap.s = 1; } break;
                case SDLK_a: { keymap.a = 1; } break;
                case SDLK_d: { keymap.d = 1; } break;
                }
            } break;
            case SDL_KEYUP: {
                switch (event.key.keysym.sym) {
                case SDLK_w: { keymap.w = 0; } break;
                case SDLK_s: { keymap.s = 0; } break;
                case SDLK_a: { keymap.a = 0; } break;
                case SDLK_d: { keymap.d = 0; } break;
                }
            } break;
            }
        }

        if (keymap.w) { camera_pos.y -= 1.0f; }
        if (keymap.s) { camera_pos.y += 1.0f; }
        if (keymap.a) { camera_pos.x -= 1.0f; }
        if (keymap.d) { camera_pos.x += 1.0f; }

        draw_test_image(screen_array, screen_buffer, camera_pos, SCREEN_WIDTH, SCREEN_HEIGHT);

        glBlitNamedFramebuffer(framebuffer, 
                               0, 0, 0, 
                               SCREEN_WIDTH, SCREEN_HEIGHT, 0, 
                               SCREEN_HEIGHT, SCREEN_WIDTH, 0, 
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        SDL_GL_SwapWindow(window);
    }

    cudaGraphicsUnregisterResource(graphics_resource);
    glDeleteRenderbuffers(1, &renderbuffer);
    glDeleteFramebuffers(1, &framebuffer);

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
