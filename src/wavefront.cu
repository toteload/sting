#include "dab/dab.h"
#include "stingmath.h"
#include "camera.h"
#include "wavefront.h"
#include "bvh.h"
#include "bvh_intersect.cu"

#define PRIME0 100030001
#define PRIME1 396191693

// Basically, you double buffer the path states
// Another approach where you don't need to double buffer is, where you keep
// one buffer of path states and then have another array that indexes into
// the path state array. The benefit of this is that you use less memory
// and you don't have to immediately write the result away when a path
// finishes. The downside is that memory access is no longer linear over the PathStates.
// Currently, I am not sure which one would be better.

// At the moment the second approach is implemented, with the indexing into the pathstate array.

extern "C"
__global__ void reset(wavefront::State* state, u32 current) {
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id != 0) {
        return;
    }

    state->total_ray_count += state->job_count[current];
    state->jobi = 0;
    state->job_count[current ^ 1] = 0;
}

extern "C"
__global__ void generate_primary_rays(wavefront::State* state, u32 current, 
                                      PointCamera camera, 
                                      u32 width, u32 height, 
                                      u32 framenum) 
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
    pathstate.emission = vec3(0.0f, 0.0f, 0.0f);
    pathstate.throughput = vec3(1.0f, 1.0f, 1.0f);
    pathstate.ray_pos = primary_ray.pos;
    pathstate.ray_dir = primary_ray.dir;

    if (id == 0) {
        state->job_count[current] = width * height;
    }
}

// !!! BROKEN !!!
extern "C"
__global__ void extend_rays_compressed_fetch(wavefront::State* state, u32 current,
                                             CBVHData cbvh, CBVHNode const * cbvh_nodes,
                                             Vector4 const * tri_pos) 
{
    constexpr u32 BLOCK_SIZE = 32;
    __shared__ u32 stack_mem[24 * BLOCK_SIZE];
    u32* const stack = stack_mem + threadIdx.x * 24;

    __shared__ u32 index_base;

    if (threadIdx.x == 0) { 
        index_base = atomicAdd(&state->jobi, BLOCK_SIZE); 
    }

    __syncthreads();

    u32 id = index_base + threadIdx.x;
    
    while (true) {
        if (id >= state->job_count[current]) {
            return;
        }

        wavefront::PathState& pathstate = state->states[state->index[current][id]];

        // Initialize the stack with the root
        stack[0] = 0;
        u32 stack_top = 1;

        BVHTriangleIntersection best;
        best.t = FLT_MAX;
        best.id = BVH_NO_HIT;
    
        const Vector3& ray_pos     = pathstate.ray_pos;
        const Vector3  ray_inv_dir = pathstate.ray_dir.inverse();
         
        while (true) {
            const bool terminated = stack_top == 0;

            const u32 mask_terminated = __ballot_sync(__activemask(), terminated);
            const u32 num_terminated  = __popc(mask_terminated);
            const u32 idx_terminated  = __popc(mask_terminated & ((1 << threadIdx.x) - 1));

            if (terminated) { 
                if (idx_terminated == 0) {
                    index_base = atomicAdd(&state->jobi, num_terminated);
                }
                __syncthreads();
                id = index_base + idx_terminated;
                
                break; 
            }

            const CBVHNode& node = cbvh_nodes[stack[--stack_top]];
            const u32 id = (node.meta & 0x0fffffff);

            if ((node.meta & 0xf0000000) != 0) {
                for (u32 i = 0; i < (node.meta >> 28); i++) {
                    const auto isect = triangle_intersect(Ray(ray_pos, pathstate.ray_dir), 
                                                          vec3(tri_pos[(id + i) * 3    ]), 
                                                          vec3(tri_pos[(id + i) * 3 + 1]), 
                                                          vec3(tri_pos[(id + i) * 3 + 2]));
                    if (isect.hit && isect.t < best.t) {
                        best.id = id + i;
                        best.t  = isect.t;
                        best.u  = isect.u;
                        best.v  = isect.v;
                    }
                }
            } else {
                f32 t_left, t_right;
                const u32 hit_left  = cbvh_bounds(cbvh, cbvh_nodes[id  ]).intersect(ray_pos, ray_inv_dir, &t_left);
                const u32 hit_right = cbvh_bounds(cbvh, cbvh_nodes[id+1]).intersect(ray_pos, ray_inv_dir, &t_right);
            
                const u32 hit_count = cast(u32, hit_left) + cast(u32, hit_right);

                if (hit_left && hit_right) {
                    const u32 right_first = cast(u32, t_left > t_right);
                    stack[stack_top  ] = id + 1 - right_first;
                    stack[stack_top+1] = id +     right_first;
                } else {
                    stack[stack_top] = id + cast(u32, hit_right);
                }

                stack_top += hit_count;
            }
        }

        pathstate.t           = best.t;
        pathstate.u           = best.u;
        pathstate.v           = best.v;
        pathstate.triangle_id = best.id;
    }
}
                                               

extern "C"
__global__ void extend_rays(wavefront::State* state, u32 current,
                            BVHNode const * bvh, 
                            Vector4 const * triangles)
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
                                       Vector4 const * triangles)
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
                      Vector4 const * triangles) 
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= state->job_count[current]) {
        return;
    }

    wavefront::PathState& pathstate = state->states[state->index[current][id]];

    RngXor32& rng = pathstate.rng;

    if (!pathstate.hit()) {
        pathstate.emission = pathstate.throughput;
        return;
    }

    const Vector3 n = triangle_normal(vec3(triangles[pathstate.triangle_id * 3    ]),
                                      vec3(triangles[pathstate.triangle_id * 3 + 1]),
                                      vec3(triangles[pathstate.triangle_id * 3 + 2]));
    const Vector3 scatter_sample = sample_cosine_weighted_hemisphere(rng.random_f32(), rng.random_f32());

    Vector3 t, b;
    build_orthonormal_basis(n, &t, &b);
    const Vector3 scatter_direction = to_world_space(scatter_sample, n, t, b);

    const Vector3 p = pathstate.ray_pos + pathstate.t * pathstate.ray_dir;
    const Ray extend_ray = Ray(p + scatter_direction * 0.001f, scatter_direction);

    const Vector3& brdf = vec3(1.0f, 1.0f, 1.0f);

    const u32 nextid = atomicAdd(&state->job_count[current ^ 1], 1);

    state->index[current ^ 1][nextid] = state->index[current][id];

    pathstate.ray_pos     = extend_ray.pos;
    pathstate.ray_dir     = extend_ray.dir;
    pathstate.throughput *= brdf;
}

extern "C"
__global__ void output_to_buffer(wavefront::State* state, Vector4* buffer, u32 width, u32 height) {
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= (width * height)) {
        return;
    }

    buffer[id] = vec4(state->states[id].emission, 1.0f);
}
