#include "dab/dab.h"
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
__global__ void reset(wavefront::State* state, u32 current) {
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id != 0) {
        return;
    }

    state->total_ray_count += state->job_count[current];
    state->job_count[current ^ 1] = 0;
}

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

    state->index[current][id] = id;

    wavefront::PathState& pathstate = state->states[id];
    pathstate.rng = rng;
    pathstate.throughput = vec3(1.0f, 1.0f, 1.0f);
    pathstate.ray_pos = primary_ray.pos;
    pathstate.ray_dir = primary_ray.dir;

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

    wavefront::PathState& pathstate = state->states[state->index[current][id]];

    const Ray ray(pathstate.ray_pos, pathstate.ray_dir);
    const BVHTriangleIntersection isect = bvh_intersect_triangles(bvh, triangles, ray);

    pathstate.t = isect.t;
    pathstate.u = isect.u;
    pathstate.v = isect.v;
    pathstate.triangle_id = isect.id;
}

extern "C"
__global__ void extend_rays_compressed(wavefront::State* state, u32 current,
                                       CBVHData cbvh, CBVHNode const * nodes,
                                       RenderTriangle const * triangles)
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= state->job_count[current]) {
        return;
    }

    wavefront::PathState& pathstate = state->states[state->index[current][id]];

    const Ray ray(pathstate.ray_pos, pathstate.ray_dir);
    const BVHTriangleIntersection isect = cbvh_intersect_triangles(cbvh, nodes, triangles, ray);

    pathstate.t = isect.t;
    pathstate.u = isect.u;
    pathstate.v = isect.v;
    pathstate.triangle_id = isect.id;
}

extern "C"
__global__ void shade(wavefront::State* state, u32 current,
                      RenderTriangle const * triangles, 
                      Material const * materials,
                      Vector4* buffer) 
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= state->job_count[current]) {
        return;
    }

    wavefront::PathState& pathstate = state->states[state->index[current][id]];
    const u32& pixel_index = state->index[current][id];

    RngXor32& rng = pathstate.rng;
    const RenderTriangle& tri = triangles[pathstate.triangle_id];

    if (!pathstate.hit()) {
        buffer[pixel_index] = vec4(pathstate.throughput, 1.0f);
        return;
    }

    const Material& material = materials[tri.material_id];

    switch (material.type) {
    case Material::DIFFUSE: {
        Vector3 n;
        {
            const Vector3 tangent_space_normal = triangle_normal_lerp(unpack_normal(tri.n0), 
                                                                      unpack_normal(tri.n1), 
                                                                      unpack_normal(tri.n2), 
                                                                      pathstate.u, pathstate.v);
            Vector3 t, b;
            build_orthonormal_basis(tri.face_normal, &t, &b);
            n = to_world_space(tangent_space_normal, tri.face_normal, t, b); 
        }

        const Vector3 scatter_sample = sample_cosine_weighted_hemisphere(rng.random_f32(), rng.random_f32());

        Vector3 t, b;
        build_orthonormal_basis(n, &t, &b);
        const Vector3 scatter_direction = to_world_space(scatter_sample, n, t, b);

        const Vector3 p = pathstate.ray_pos + pathstate.t * pathstate.ray_dir;
        const Ray extend_ray = Ray(p + scatter_direction * 0.0001f, scatter_direction);

        const Vector3& brdf = material.color();

        const u32 nextid = atomicAdd(&state->job_count[current ^ 1], 1);

        state->index[current ^ 1][nextid] = state->index[current][id];

        pathstate.ray_pos     = extend_ray.pos;
        pathstate.ray_dir     = extend_ray.dir;
        pathstate.throughput *= brdf;
    } break;
    case Material::EMISSIVE: {
        buffer[pixel_index] = vec4(pathstate.throughput * material.light_intensity * material.color(), 1.0f);
    } break;
    }
}
