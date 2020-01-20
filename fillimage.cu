#include <stdint.h>

struct vec4 {
    float x, y, z, w;

    __device__ vec4() { }
    __device__ vec4(float a): x(a), y(a), z(a), w(a) { }
};

surface<void, cudaSurfaceType2D> screen_surface;

__global__ void fill_framebuffer(uint32_t width, uint32_t height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    vec4 c;
    c.x = float(x) / width;
    c.y = float(y) / height;
    c.z = 0.0f;
    c.w = 1.0f;

    surf2Dwrite<vec4>(c, screen_surface, x * sizeof(vec4), y, cudaBoundaryModeZero);
}

extern "C" void draw_test_image(cudaArray_const_t array, uint32_t width, uint32_t height) {
    cudaBindSurfaceToArray(screen_surface, array);
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3(width / threads.x + 1, height / threads.y + 1, 1);
    fill_framebuffer<<<blocks, threads>>>(width, height);
    cudaDeviceSynchronize();
}
