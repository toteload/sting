#include <stdint.h>
#include "vecmath.h"

surface<void, cudaSurfaceType2D> screen_surface;

__global__ void fill_screen_buffer(vec4* buffer, uint32_t width, uint32_t height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int id = y * width + x;

    vec4 c = { 1.0f, 0.0f };
    c.r = float(x) / width;
    c.g = sinf(float(y)/height * 10);
    c.b = 0.0f;
    c.a = 1.0f;

    buffer[id] = c;
}

__global__ void blit_to_screen(vec4* buffer, uint32_t width, uint32_t height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int id = y * width + x;

    surf2Dwrite<vec4>(buffer[id], screen_surface, x * sizeof(vec4), y, cudaBoundaryModeZero);
}

void draw_test_image(cudaArray_const_t array, vec4* screen_buffer, uint32_t width, uint32_t height) {
    cudaBindSurfaceToArray(screen_surface, array);
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    fill_screen_buffer<<<blocks, threads>>>(screen_buffer, width, height);
    blit_to_screen<<<blocks, threads>>>(screen_buffer, width, height);
    cudaDeviceSynchronize();
}
