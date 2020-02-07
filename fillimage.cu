#include "common.h"
#include "stingmath.h"
#include "camera.h"
#include "bvh.h"
#include "bvh.cpp"

surface<void, cudaSurfaceType2D> screen_surface;

__device__ vec3 pathtrace_bruteforce(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray, 
                                     u32 seed, u32 depth = 0) 
{
    const vec3 BLACK(0.0f);

    if (depth == 4) {
        return BLACK;
    }

    float t, u, v;
    uint32_t tri_id;
    const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &u, &v, &tri_id);

    if (!hit) {
        return BLACK;
    }

    const RenderTriangle& tri = triangles[tri_id];
    const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, u, v);
    const vec3 p = ray.pos + t * ray.dir;

    switch (tri.material) {
    case MATERIAL_DIFFUSE: {
        const f32 r0 = rng_xor32(seed), r1 = rng_xor32(seed);
        const vec3 scatter_direction = diffuse_reflection(n, r0, r1);
        const Ray scatter_ray(p + scatter_direction * 0.0001f, scatter_direction);
        const vec3 brdf = (1.0f / M_PI) * tri.color;
        const vec3 ei = dot(scatter_direction, n) * pathtrace_bruteforce(bvh, triangles, scatter_ray, depth + 1);
        return M_2_PI * brdf * ei;
    } break;
    case MATERIAL_EMISSIVE: {
        return tri.light_intensity * tri.color;
    } break;
    }

    return BLACK;
}

__device__ bool intersect(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray, HitRecord* hit_out) {
    float t, u, v;
    uint32_t tri_id;
    const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &u, &v, &tri_id);

    if (!hit) {
        return false;
    }

    const RenderTriangle& tri = triangles[tri_id];
#if 0
    vec3 normal = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, u, v);
#else
    vec3 normal = triangle_normal(tri.v0, tri.v1, tri.v2);
    if (dot(normal, ray.dir) > 0.0f) {
        normal = -1.0f * normal;
    }
#endif

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
