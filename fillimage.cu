#include <stdint.h>
#include "vecmath.h"
#include "bvh.h"
#include "bvh.cpp"

surface<void, cudaSurfaceType2D> screen_surface;

__device__ bool intersect(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray, HitRecord* hit_out) {
    float t;
    uint32_t tri_id;
    uint32_t aabb_isect_count;
    uint32_t tri_isect_count;
    const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &tri_id, &aabb_isect_count, &tri_isect_count);

    if (!hit) {
        return false;
    }

    const RenderTriangle& tri = triangles[tri_id];
    vec3 normal = triangle_normal(tri.v0, tri.v1, tri.v2);
    if (dot(normal, ray.dir) > 0.0f) {
        normal = -1.0f * normal;
    }

    HitRecord rec;
    rec.pos = ray.pos + t * ray.dir;
    rec.t = t;
    rec.normal = normal;

    *hit_out = rec;

    return true;
}

__global__ void fill_screen_buffer(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
                                   vec4* buffer, uint32_t width, uint32_t height) 
{
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

    const vec3 point_light = { 0.0f, 1000.0f, 0.0f };

    vec4 c = vec4(0.0f, 0.0f, 0.0f, 1.0f);

#if 0
    /*
    // 4968
    for (uint32_t i = 0; i < 1000; i++) {
        float t;
        if (triangle_intersect(ray, triangles[i].v0, triangles[i].v1, triangles[i].v2, &t)) {
            c = vec4(1.0f, 0.0, 0.0f, 1.0f);
            break;
        }
    }
    */

    //float t;
    //uint32_t tri_id;
    //uint32_t aabb_isect_count;
    //uint32_t tri_isect_count;
    //const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &tri_id, &aabb_isect_count, &tri_isect_count);
    //float v = float(aabb_isect_count) / 80.0f;
    //float v = float(tri_isect_count) / 400.0f;
    //float v = hit ? 1.0f : 0.0f;
    c = vec4(v, v, v, 1.0f);
    //if (hit) { c = vec4(1.0f, 0.0f, 0.0f, 1.0f); }

    /*
    HitRecord rec;
    const bool hit = intersect(bvh, triangles, ray, &rec);
    if (hit) { c = vec4(1.0f, 0.0f, 0.0f, 1.0f); }
    */
#else
    HitRecord rec;
    const bool hit = intersect(bvh, triangles, ray, &rec);

    if (hit) {
        const float SHADOW_OFFSET_EPSILON = 0.0001f;

        const vec3 to_light = (point_light - rec.pos).normalize();
        const Ray shadow_ray = { rec.pos + SHADOW_OFFSET_EPSILON * rec.normal, to_light };

        const float max_distance = (point_light - rec.pos).length();

        const bool occluded = bvh_intersect_triangles_shadowcast(bvh, triangles, shadow_ray, max_distance);
        if (!occluded) {
            const float v = dot(to_light, rec.normal);
            c = vec4(v, v, v, 1.0f);
        }
    }
#endif

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

// ----------------------------------------------------------------------------

void render(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
            vec4* screen_buffer, uint32_t width, uint32_t height) 
{
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    fill_screen_buffer<<<blocks, threads>>>(bvh, triangles, camera, screen_buffer, width, height);
}

void render_buffer_to_screen(cudaArray_const_t array, vec4* screen_buffer, uint32_t width, uint32_t height) {
    const dim3 threads = dim3(16, 16, 1);
    const dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);

    cudaBindSurfaceToArray(screen_surface, array);
    blit_to_screen<<<blocks, threads>>>(screen_buffer, width, height);

    // Need to synchronize here otherwise it is very choppy
    cudaDeviceSynchronize();
}
