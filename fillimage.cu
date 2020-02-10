#include "common.h"
#include "stingmath.h"
#include "camera.h"
#include "bvh.h"
#include "bvh.cpp"

surface<void, cudaSurfaceType2D> screen_surface;

#define PRIME0 100030001
#define PRIME1 396191693

__global__ void normal_test_pass(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera,
                                 vec4* buffer, u32 width, u32 height, u32 framenum)
{
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;    
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;    

    if (x >= width || y >= height) {
        return;
    }

    const u32 id = y * width + x;

    // nx and ny are in range (-1.0f, 1.0f)
    const float nx = (2.0f * float(x) + 0.5f) / width  - 1.0f;
    const float ny = (2.0f * float(y) + 0.5f) / height - 1.0f;

    const Ray ray = camera.create_ray(nx, ny);

    float t, u, v;
    uint32_t tri_id;
    const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &u, &v, &tri_id);
    if (!hit) { 
        buffer[id] = vec4(0.0f, 0.0f, 0.0f, 1.0f); 
        return; 
    }

    const RenderTriangle& tri = triangles[tri_id];
    const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, u, v);

    buffer[id] = vec4(n, 1.0f);
}

__device__ vec3 pathtrace_bruteforce(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray, u32 seed,
                                     vec4 const * skybox) 
{
    const vec3 BLACK(0.0f);

    vec3 acc = vec3(1.0f);

    for (u32 depth = 0; ; depth++) {
        if (depth == 3) {
            return BLACK;
        }

        float t, u, v;
        uint32_t tri_id;
        const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &u, &v, &tri_id);

        if (!hit) {
            //Sample the skybox
            f32 inclination, azimuth;
            cartesian_to_spherical(ray.dir, &inclination, &azimuth);
            const u32 ui = __float2uint_rd((azimuth / (2.0f * M_PI) + 0.5f) * 4095.0f + 0.5f);
            const u32 vi = __float2uint_rd((inclination / M_PI) * 2047.0f + 0.5f);
            const vec4 sky = skybox[vi * 4096 + ui];
            return vec3(sky.r, sky.g, sky.b);
        }

        const RenderTriangle& tri = triangles[tri_id];

        switch (tri.material) {
        case MATERIAL_DIFFUSE: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, u, v);
            const vec3 p = ray.pos + t * ray.dir;
            const f32 r1 = rng_xor32(seed); 
            const f32 r2 = rng_xor32(seed);
            const vec3 scatter_sample = sample_uniform_hemisphere(r1, r2);
            vec3 t, b;
            build_orthonormal_basis(n, &t, &b);
            const vec3 scatter_direction = to_world_space(scatter_sample, n, t, b);
            const vec3 color = tri.color;
            const vec3 brdf = (1.0f / M_PI) * color;
            acc *= 2.0f * M_PI * brdf * dot(scatter_direction, n);

            // Set the ray for the next loop iteration
            ray = Ray(p + scatter_direction * 0.0001f, scatter_direction);
        } break;
        case MATERIAL_EMISSIVE: {
            if (depth == 0) { 
                return tri.light_intensity * tri.color; 
            } else { 
                return acc * tri.light_intensity * tri.color; 
            }
        } break;
        case MATERIAL_MIRROR: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, u, v);
            const vec3 p = ray.pos + t * ray.dir;
            const vec3 reflection = reflect(n, ray.dir);
            ray = Ray(p + reflection * 0.0001f, reflection);
        } break;
        }
    }
}

__device__ bool intersect(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray, HitRecord* hit_out) {
    float t, u, v;
    uint32_t tri_id;
    const bool hit = bvh_intersect_triangles(bvh, triangles, ray, &t, &u, &v, &tri_id);

    if (!hit) {
        return false;
    }

    const RenderTriangle& tri = triangles[tri_id];
    vec3 normal = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, u, v);

    HitRecord rec;
    rec.pos = ray.pos + t * ray.dir;
    rec.t = t;
    rec.normal = normal;

    *hit_out = rec;

    return true;
}

__global__ void accumulate_pass(vec4* frame_buffer, vec4* accumulator, vec4* screen_buffer, 
                                u32 width, u32 height, u32 acc_frame) 
{
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;    
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;    

    if (x >= width || y >= height) {
        return;
    }

    const u32 id = y * width + x;

    if (acc_frame == 0) {
        accumulator[id] = 0.0f;
    }

    accumulator[id] += frame_buffer[id];
    screen_buffer[id] = accumulator[id] / (cast(f32, acc_frame + 1));
}

__global__ void test_001(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera,
                         vec4 const * skybox,
                         vec4* buffer, u32 width, u32 height, u32 framenum)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int id = y * width + x;
    u32 seed = (id + framenum * PRIME0) * PRIME1;

    // nx and ny are in range (-1.0f, 1.0f)
    const float nx = (2.0f * float(x) + rng_xor32(seed)) / width  - 1.0f;
    const float ny = (2.0f * float(y) + rng_xor32(seed)) / height - 1.0f;

    const Ray ray = camera.create_ray(nx, ny);

    const vec3 c = pathtrace_bruteforce(bvh, triangles, ray, seed, skybox);
    
    buffer[id] = vec4(c, 1.0f);
}

__global__ void test_000(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
                         vec4* buffer, uint32_t width, uint32_t height, u32 framenum) 
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
void accumulate(vec4* frame_buffer, vec4* accumulator, vec4* screen_buffer, u32 width, u32 height, u32 acc_frame) {
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    accumulate_pass<<<blocks, threads>>>(frame_buffer, accumulator, screen_buffer, width, height, acc_frame);
}

void render(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
            vec4 const * skybox,
            vec4* buffer, u32 width, u32 height, u32 framenum) 
{
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    test_001<<<blocks, threads>>>(bvh, triangles, camera, skybox, buffer, width, height, framenum);
}

void render_normal(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
                   vec4* buffer, u32 width, u32 height, u32 framenum) 
{
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    normal_test_pass<<<blocks, threads>>>(bvh, triangles, camera, buffer, width, height, framenum);
}

void render_buffer_to_screen(cudaArray_const_t array, vec4* screen_buffer, uint32_t width, uint32_t height) {
    const dim3 threads = dim3(16, 16, 1);
    const dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);

    cudaBindSurfaceToArray(screen_surface, array);
    blit_to_screen<<<blocks, threads>>>(screen_buffer, width, height);

    // Need to synchronize here otherwise it is very choppy
    cudaDeviceSynchronize();
}
