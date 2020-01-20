#include "ext/SDL2-2.0.10/include/SDL.h"
#include "common.h"
#include <Windows.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include "cuda_gl_interop.h"

int main(int argc, char** args) {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("sting",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          1080,
                                          540,
                                          0);

    int running = 1;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT: running = 0; break;
            case SDL_KEYDOWN: {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = 0;
                }
            }
            }
        }


    }

    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
