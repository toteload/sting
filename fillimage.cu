#include <stdint.h>
#include "vecmath.h"

surface<void, cudaSurfaceType2D> screen_surface;

__global__ void fill_screen_buffer(PointCamera camera, vec4* buffer, uint32_t width, uint32_t height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int id = y * width + x;

    // nx and ny are in range (-1.0f, 1.0f)
    const float nx = (2.0f * float(x) + 0.5f) / width  - 1.0f;
    const float ny = (2.0f * float(y) + 0.5f) / height - 1.0f;

    Ray ray = camera.create_ray(nx, ny);

    // Do ray sphere intersection

    Sphere spheres[4] = {
        { { 0.0f, 0.0f, -200.0f }, 100.f },
        { { 0.0f, 0.0f,  200.0f }, 100.f },
        { { 0.0f, 100.0f, -200.0f }, 100.f },
        { { 100.0f, 0.0f, -200.0f }, 100.f },
    };

    HitRecord hitrecord;
    uint32_t hit = 0;
    for (int i = 0; i < 4; i++) {
        HitRecord record;
        uint32_t sphere_hit = spheres[i].intersect(ray, &record);
        if (sphere_hit && (!hit || record.t < hitrecord.t)) { 
            hit = 1; 
            hitrecord = record;
        }
    }

    vec4 c;
    if (hit) {
        c = { hitrecord.normal.x, hitrecord.normal.y, hitrecord.normal.z, 1.0f };
    } else {
        const vec4 black = { 0.0f, 0.0f, 0.0f, 1.0f };
        c = black;
    }

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

void fill_buffer(vec4* screen_buffer, PointCamera camera, uint32_t width, uint32_t height) {
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    fill_screen_buffer<<<blocks, threads>>>(camera, screen_buffer, width, height);
}

void render_buffer_to_screen(cudaArray_const_t array, vec4* screen_buffer, uint32_t width, uint32_t height) {
    const dim3 threads = dim3(16, 16, 1);
    const dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);

    cudaBindSurfaceToArray(screen_surface, array);
    blit_to_screen<<<blocks, threads>>>(screen_buffer, width, height);
}
