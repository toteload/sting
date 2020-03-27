#include "common.h"
#include "stingmath.h"
#include "camera.h"
#include "bvh.h"
#include "bvh.cpp"

surface<void, cudaSurfaceType2D> screen_surface;

#define PRIME0 100030001
#define PRIME1 396191693

__device__ vec3 normal_test_pass(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray) {
    const auto isect = bvh_intersect_triangles(bvh, triangles, ray);
    if (!isect.hit()) { 
        return vec3(0.0f, 0.0f, 0.0f); 
    }

    const RenderTriangle& tri = triangles[isect.id];
    const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
    return n;
}

// path tracing with next event estimation
__device__ vec3 pathtrace_nee(BVHNode const * bvh, RenderTriangle const * triangles, u32 const * lights,
                              u32 light_count, Ray ray, RngXor32 rng, vec4 const * skybox)
{
    vec3 emission = vec3(0.0f, 0.0f, 0.0f);
    vec3 throughput = vec3(1.0f, 1.0f, 1.0f);
    bool allow_emissive = true;

    for (u32 depth = 0; depth < 2; depth++) {
        const BVHTriangleIntersection isect = bvh_intersect_triangles(bvh, triangles, ray);

        if (!isect.hit()) {
            //Sample the skybox
            f32 inclination, azimuth;
            cartesian_to_spherical(ray.dir, &inclination, &azimuth);

            // dimensions of the skybox are hardcoded
            // float to uint round down
            const u32 ui = __float2uint_rd((azimuth / (2.0f * M_PI) + 0.5f) * 4095.0f + 0.5f);
            const u32 vi = __float2uint_rd((inclination / M_PI) * 2047.0f + 0.5f);
            const vec4 sky = skybox[vi * 4096 + ui];
            emission += throughput * vec3(sky.r, sky.g, sky.b);
            break;
        }

        const RenderTriangle& tri = triangles[isect.id];

        switch (tri.material) {
        case MATERIAL_DIFFUSE: {
            allow_emissive = false;

            // sample light directly
            // ----------------------------------------------------------------
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
            const vec3 p = ray.pos + isect.t * ray.dir;

            const u32 random_light_id = lights[rng.random_u32_max(light_count)];
            const RenderTriangle& light = triangles[random_light_id];
            const vec3 point_on_light = triangle_random_point(light.v0, light.v1, light.v2, 
                                                              rng.random_f32(), rng.random_f32());
            const vec3 dir_to_light = (point_on_light - p).normalize();

            // angle between the normal of the surface of the light and the ray
            // this is only used if you want your emissive triangles to only
            // emit light from one side.
            //const vec3 light_normal = light.face_normal;
            //const f32 coso = dot(-1.0f * dir_to_light, light_normal);

            // angle between the normal of the diffuse surface and the ray towards the light
            const f32 cosi = dot(dir_to_light, n); 

            if (cosi > 0.0f) {
                const Ray shadow_ray(p + 0.0001f * n, dir_to_light);
                const auto light_isect = bvh_intersect_triangles(bvh, triangles, shadow_ray);

                if (light_isect.hit() && light_isect.id == random_light_id) {
                    const vec3 brdf = tri.color();

                    // This does correctly calculate the solid angle of the triangle,
                    // but does not take into account how much of the triangle might
                    // not be visible for the shading hemisphere. I think this
                    // gets corrected though, due to the fact that those parts
                    // of the triangle cannot get hit.
                    const f32 solid_angle_estimation = fabs(triangle_solid_angle(light.v0 - p, 
                                                                                 light.v1 - p, 
                                                                                 light.v2 - p));

                    // the probability of hitting this light is the proportion of the hemisphere
                    // covered by this light (the solid angle)
                    const f32 light_pdf = 1.0f / solid_angle_estimation;
                    emission += throughput * (cosi / light_pdf) * brdf * light.color() * light.light_intensity;
                }
            }

            // GI bounce
            // ----------------------------------------------------------------
            const vec3 scatter_sample = sample_cosine_weighted_hemisphere(rng.random_f32(), rng.random_f32());

            vec3 t, b;
            build_orthonormal_basis(n, &t, &b);
            const vec3 scatter_direction = to_world_space(scatter_sample, n, t, b);

            ray = Ray(p + 0.0001f * scatter_direction, scatter_direction);

            throughput *= tri.color();
        } break;
        case MATERIAL_EMISSIVE: {
            if (allow_emissive) {
                emission += throughput * tri.light_intensity * tri.color();
            }

            return emission;
        } break;
        case MATERIAL_MIRROR: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
            const vec3 p = ray.pos + isect.t * ray.dir;
            const vec3 reflection = reflect(n, ray.dir);
            ray = Ray(p + reflection * 0.0001f, reflection);   
            allow_emissive = true;
        } break;
        }
    }

    return emission;
}

// using cosine weighted hemisphere sampling
__device__ vec3 pathtrace_bruteforce_2(BVHNode const * bvh, RenderTriangle const * triangles, 
                                       Ray ray, RngXor32 rng, vec4 const * skybox) 
{
    vec3 emission = vec3(0.0f);
    vec3 throughput = vec3(1.0f);

    for (u32 depth = 0; depth < 3; depth++) {
        const BVHTriangleIntersection isect = bvh_intersect_triangles(bvh, triangles, ray);

        if (!isect.hit()) {
#if 0
            emission += throughput;
            break;
#else
            //Sample the skybox
            f32 inclination, azimuth;
            cartesian_to_spherical(ray.dir, &inclination, &azimuth);
            // float to uint round down
            const u32 ui = __float2uint_rd((azimuth / (2.0f * M_PI) + 0.5f) * 4095.0f + 0.5f);
            const u32 vi = __float2uint_rd((inclination / M_PI) * 2047.0f + 0.5f);
            const vec4 sky = skybox[vi * 4096 + ui];
            emission += throughput * vec3(sky.r, sky.g, sky.b);
            break;
#endif
        }

        const RenderTriangle& tri = triangles[isect.id];

        switch (tri.material) {
        case MATERIAL_DIFFUSE: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
            const vec3 p = ray.pos + isect.t * ray.dir;

            const vec3 scatter_sample = sample_cosine_weighted_hemisphere(rng.random_f32(), rng.random_f32());

            vec3 t, b;
            build_orthonormal_basis(n, &t, &b);
            const vec3 scatter_direction = to_world_space(scatter_sample, n, t, b);

            const vec3 brdf = tri.color();

            throughput *= brdf;

            // Set the ray for the next loop iteration
            ray = Ray(p + scatter_direction * 0.0001f, scatter_direction);
        } break;
        case MATERIAL_EMISSIVE: {
            emission += throughput * tri.light_intensity * tri.color();
            return emission;
        } break;
        case MATERIAL_MIRROR: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
            const vec3 p = ray.pos + isect.t * ray.dir;
            const vec3 reflection = reflect(n, ray.dir);
            ray = Ray(p + reflection * 0.0001f, reflection);
        } break;
        }
    }

    return emission;
}

__device__ vec3 pathtrace_bruteforce(BVHNode const * bvh, RenderTriangle const * triangles, 
                                     Ray ray, RngXor32 rng, vec4 const * skybox) 
{
    vec3 acc = vec3(1.0f);

    for (u32 depth = 0; ; depth++) {
        if (depth == 3) {
            return vec3(0.0f);
        }

        const auto isect = bvh_intersect_triangles(bvh, triangles, ray);

        if (!isect.hit()) {
            //Sample the skybox
            f32 inclination, azimuth;
            cartesian_to_spherical(ray.dir, &inclination, &azimuth);
            const u32 ui = __float2uint_rd((azimuth / (2.0f * M_PI) + 0.5f) * 4095.0f + 0.5f);
            const u32 vi = __float2uint_rd((inclination / M_PI) * 2047.0f + 0.5f);
            const vec4 sky = skybox[vi * 4096 + ui];
            return acc * vec3(sky.r, sky.g, sky.b);
        }

        const RenderTriangle& tri = triangles[isect.id];

        switch (tri.material) {
        case MATERIAL_DIFFUSE: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
            const vec3 p = ray.pos + isect.t * ray.dir;

            const vec3 scatter_sample = sample_uniform_hemisphere(rng.random_f32(), rng.random_f32());

            vec3 t, b;
            build_orthonormal_basis(n, &t, &b);
            const vec3 scatter_direction = to_world_space(scatter_sample, n, t, b);

#if 1
            const vec3 brdf = (1.0f / M_PI) * tri.color();
            acc *= 2.0f * M_PI * brdf * dot(scatter_direction, n);
#else
            const vec3 brdf = tri.color();
            acc *= 2.0f * color * dot(scatter_direciton, n);
#endif

            // Set the ray for the next loop iteration
            ray = Ray(p + scatter_direction * 0.0001f, scatter_direction);
        } break;
        case MATERIAL_EMISSIVE: {
            return acc * tri.light_intensity * tri.color(); 
        } break;
        case MATERIAL_MIRROR: {
            const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
            const vec3 p = ray.pos + isect.t * ray.dir;
            const vec3 reflection = reflect(n, ray.dir);
            ray = Ray(p + reflection * 0.0001f, reflection);
        } break;
        }
    }
}

extern "C"
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
        accumulator[id] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    accumulator[id] += frame_buffer[id];
    screen_buffer[id] = accumulator[id] / (cast(f32, acc_frame + 1));
}

__global__ void intersect_test(BVHNode const * bvh, RenderTriangle const * triangles, 
                               PointCamera camera, vec4* buffer, u32 width, u32 height)
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

    const Ray ray = camera.create_ray(nx, ny);

    const auto isect = bvh_intersect_triangles(bvh, triangles, ray);

    if (isect.hit()) {
        buffer[id] = vec4(vec3(1.0f), 1.0f);
    } else {
        buffer[id] = vec4(vec3(0.0f), 1.0f);
    }
}

extern "C"
__global__ void nee_test(BVHNode const * bvh, RenderTriangle const * triangles, 
                         u32 const * lights, u32 light_count,
                         PointCamera camera, vec4 const * skybox,
                         vec4* buffer, u32 width, u32 height, u32 framenum)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int id = y * width + x;
    const u32 seed = (id + framenum * PRIME0) * PRIME1;
    RngXor32 rng(seed);

    // nx and ny are in range (-1.0f, 1.0f)
    const float nx = (2.0f * float(x) + rng.random_f32()) / width  - 1.0f;
    const float ny = (2.0f * float(y) + rng.random_f32()) / height - 1.0f;

    const Ray ray = camera.create_ray(nx, ny);

    const vec3 c = pathtrace_nee(bvh, triangles, lights, light_count, ray, rng, skybox);
    
    buffer[id] = vec4(c, 1.0f);
}

extern "C"
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

    const u32 seed = (id + framenum * PRIME0) * PRIME1;
    RngXor32 rng(seed);

    // nx and ny are in range (-1.0f, 1.0f)
    const float nx = (2.0f * float(x) + rng.random_f32()) / width  - 1.0f;
    const float ny = (2.0f * float(y) + rng.random_f32()) / height - 1.0f;

    const Ray ray = camera.create_ray(nx, ny);

    const vec3 c = pathtrace_bruteforce_2(bvh, triangles, ray, rng, skybox);

    buffer[id] = vec4(c, 1.0f);
}

extern "C"
__global__ void blit_to_screen(vec4* buffer, uint32_t width, uint32_t height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int id = y * width + x;

#if 1
    surf2Dwrite<vec4>(buffer[id], screen_surface, x * sizeof(vec4), y, cudaBoundaryModeZero);
#else
    surf2Dwrite<vec4>(vec4(1.0f, 0.0f, 0.0f, 1.0f), screen_surface, x * sizeof(vec4), y, cudaBoundaryModeZero);
#endif
}

// ----------------------------------------------------------------------------
#if 0
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
    //intersect_test<<<blocks, threads>>>(bvh, triangles, camera, buffer, width, height);
}

void render_nee(BVHNode const * bvh, RenderTriangle const * triangles, 
                u32 const * lights, u32 light_count,
                PointCamera camera, vec4 const * skybox,
                vec4* buffer, u32 width, u32 height, u32 framenum) 
{
    dim3 threads = dim3(16, 16, 1);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
    nee_test<<<blocks, threads>>>(bvh, triangles, lights, light_count, camera, skybox, buffer, width, height, framenum);
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
#endif
