#include "common.h"
#include "stingmath.h"
#include "camera.h"
#include "wavefront.h"
#include "bvh.h"
#include "bvh.cpp"

#define PRIME0 100030001
#define PRIME1 396191693

// Basically, you double buffer the path states
// Another approach where you don't need to double buffer is, where you keep
// one buffer of path states and then have another array that indexes into
// the path state array. The benefit of this is that you use less memory
// and you don't have to immediately write the result away when a path
// finishes. The downside is that memory access is no longer linear.
// Currently, I am not sure which one would be better.

extern "C"
__global__ void generate_primary_rays(wavefront::State* state, u32 current, 
                                      PointCamera camera, u32 width, u32 height, u32 framenum) 
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= (width * height)) {
        return;
    }

    const u32 x = id % width;
    const u32 y = id / width;

    const u32 seed = (id + framenum * PRIME0) * PRIME1;
    RngXor32 rng(seed);

    // nx and ny are in range (-1.0f, 1.0f)
    const f32 nx = (2.0f * cast(f32, x) + rng.random_f32()) / width  - 1.0f;
    const f32 ny = (2.0f * cast(f32, y) + rng.random_f32()) / height - 1.0f;

    const Ray primary_ray = camera.create_ray(nx, ny);

    state->states[current][id].rng = rng;
    state->states[current][id].throughput = vec3(1.0f, 1.0f, 1.0f);
    state->states[current][id].ray_pos = primary_ray.pos;
    state->states[current][id].ray_dir = primary_ray.dir;
    state->states[current][id].pixel_index = id;

    if (id == 0) {
        state->job_count[current] = width * height;
    }
}

extern "C"
__global__ void extend_rays(wavefront::State* state, u32 current,
                            BVHNode const * bvh, RenderTriangle const * triangles)
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= state->job_count[current]) {
        return;
    }

    const Ray ray(state->states[current][id].ray_pos, state->states[current][id].ray_dir);
    const BVHTriangleIntersection isect = bvh_intersect_triangles(bvh, triangles, ray);

    state->states[current][id].t = isect.t;
    state->states[current][id].u = isect.u;
    state->states[current][id].v = isect.v;
    state->states[current][id].triangle_id = isect.id;

#if 1
    if (id == 0) {
        state->job_count[current ^ 1] = 0;
    }
#endif
}

extern "C"
__global__ void shade(wavefront::State* state, u32 current,
                      RenderTriangle const * triangles, vec4* buffer) 
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= state->job_count[current]) {
        return;
    }

    wavefront::PathState& pathstate = state->states[current][id];

    RngXor32& rng = pathstate.rng;
    const RenderTriangle& tri = triangles[pathstate.triangle_id];

    if (!pathstate.hit()) {
        buffer[pathstate.pixel_index] = vec4(pathstate.throughput, 1.0f);
        return;
    }

    switch (tri.material) {
    case MATERIAL_DIFFUSE: {
        const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, pathstate.u, pathstate.v);
        const vec3 scatter_sample = sample_cosine_weighted_hemisphere(rng.random_f32(), rng.random_f32());

        vec3 t, b;
        build_orthonormal_basis(n, &t, &b);
        const vec3 scatter_direction = to_world_space(scatter_sample, n, t, b);

        const vec3 p = pathstate.ray_pos + pathstate.t * pathstate.ray_dir;
        const Ray extend_ray = Ray(p + scatter_direction * 0.0001f, scatter_direction);

        const u32 nextid = atomicAdd(&state->job_count[current ^ 1], 1);

        state->states[current ^ 1][nextid].ray_pos = extend_ray.pos;
        state->states[current ^ 1][nextid].ray_dir = extend_ray.dir;
        state->states[current ^ 1][nextid].rng = pathstate.rng;
        state->states[current ^ 1][nextid].pixel_index = pathstate.pixel_index;

        const vec3& brdf = tri.color();
        state->states[current ^ 1][nextid].throughput = pathstate.throughput * brdf;
    } break;
    case MATERIAL_EMISSIVE: {
        buffer[pathstate.pixel_index] = vec4(pathstate.throughput * tri.light_intensity * tri.color(), 1.0f);
    } break;
    }
}
