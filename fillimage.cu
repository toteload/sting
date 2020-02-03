#include <stdint.h>
#include "vecmath.h"

surface<void, cudaSurfaceType2D> screen_surface;

__device__ void intersect(Sphere const * spheres, uint32_t sphere_count, 
                          Triangle const * triangles, uint32_t triangle_count,
                          Ray ray, HitRecord* out) 
{
    HitRecord rec = HitRecord::no_hit();

    for (uint32_t i = 0; i < sphere_count; i++) {
        HitRecord r;
        spheres[i].intersect(ray, &r);
        if (r.hit && (!rec.hit || r.t < rec.t)) {
            rec = r;
        }
    }

    for (uint32_t i = 0; i < triangle_count; i++) {
        HitRecord r;
        triangles[i].intersect(ray, &r);
        if (r.hit && (!rec.hit || r.t < rec.t)) {
            rec = r;
        }
    }

    *out = rec;
}

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

    const vec3 point_light = { 0.0f, 1000.0f, 0.0f };

    const Sphere spheres[] = {
        { { 0.0f, 0.0f, -200.0f }, 100.f },
        { { 0.0f, 0.0f,  200.0f }, 100.f },
        { { 0.0f, 100.0f, -200.0f }, 100.f },
        { { 100.0f, 0.0f, -200.0f }, 100.f },
    };

    const Triangle triangles[] = {
        { { -1000.0f, 0.0f, -1000.0f }, { 1000.0f, 0.0f, -1000.0f }, { 1000.0f, 0.0f, 1000.0f } },
        { { -1000.0f, 0.0f, -1000.0f }, { -1000.0f, 0.0f, 1000.0f }, { 1000.0f, 0.0f, 1000.0f } },
    };

    HitRecord rec;
    intersect(spheres, 4, triangles, 2, ray, &rec);

    const vec4 black = { 0.0f, 0.0f, 0.0f, 1.0f };

    vec4 c;
    if (rec.hit) {
        const float EPSILON = 0.00001f; // just some value

        const vec3 to_light = (point_light - rec.pos).normalize();

        Ray shadow_ray = { rec.pos + EPSILON * rec.normal, to_light };

        HitRecord shadow_rec;
        intersect(spheres, 4, triangles, 2, shadow_ray, &shadow_rec);

        const uint32_t is_occluded = shadow_rec.hit && ((rec.pos - point_light).length() > shadow_rec.t);

        if (is_occluded) {
            c = { 0.0f, 0.0f, 0.0f, 1.0f };
        } else {
            const float v = dot(to_light, rec.normal);
            c = { v, v, v, 1.0f };
        }
    } else {
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
