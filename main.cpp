#include "ext/SDL2-2.0.10/include/SDL.h"
#include <stdio.h>
#include "common.h"
#include <Windows.h>
#include <GL/gl.h>
#include "glext.h"
#include "wglext.h"
#include "ogl_extra.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

const u32 SCREEN_WIDTH = 1080;
const u32 SCREEN_HEIGHT = 540;

struct cuda_gl_interop {
    u32 width;
    u32 height;

    GLuint framebuffer;
    GLuint renderbuffer;

    cudaGraphicsResource* g;
    cudaArray* a;

    cudaError create(u32 width, u32 height);
    void destroy();
    void blit();
};

cudaError cuda_gl_interop::create(u32 width, u32 height) {
    this->width = width;
    this->height = height;

    glCreateRenderbuffers(1, &renderbuffer);
    glCreateFramebuffers(1, &framebuffer);

    glNamedFramebufferRenderbuffer(framebuffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);

    glNamedRenderbufferStorage(renderbuffer, GL_RGBA32F, width, height);

    cudaError err = cudaSuccess;
    err = cudaGraphicsGLRegisterImage(&g, 
                                      renderbuffer, GL_RENDERBUFFER, 
                                      cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
    err = cudaGraphicsMapResources(1, &g, 0);
    err = cudaGraphicsSubResourceGetMappedArray(&a, g, 0, 0);
    err = cudaGraphicsUnmapResources(1, &g, 0);

    return err;
}

void cuda_gl_interop::destroy() {
    // TODO
}

void cuda_gl_interop::blit() {
    glBlitNamedFramebuffer(framebuffer, 0, 0, 0, width, height, 0, height, width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

extern "C" void draw_test_image(cudaArray_const_t, uint32_t, uint32_t);

int main(int argc, char** args) {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
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
        printf("Failed to create gl context\n");
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);

    SDL_GL_SetSwapInterval(0);

    if (!ogl_extra_init()) {
        printf("Failed to get some gl extensions\n");
        return 1;
    }

    float* blitbuffer;
    cudaMalloc(&blitbuffer, SCREEN_WIDTH * SCREEN_HEIGHT * 4 * sizeof(float));

    cuda_gl_interop interop;
    interop.create(SCREEN_WIDTH, SCREEN_HEIGHT);

    int running = 1;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT: { running = 0; } break;
            case SDL_KEYDOWN: {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = 0;
                }
            } break;
            }
        }

        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        draw_test_image(interop.a, SCREEN_WIDTH, SCREEN_HEIGHT);
        interop.blit();

        SDL_GL_SwapWindow(window);
    }

    interop.destroy();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
