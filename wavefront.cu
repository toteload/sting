#define PRIME0 100030001
#define PRIME1 396191693

#if 0
struct WavefrontState {
    PathState* state_buffer[2]; // device memory
    StateCounter* counter; // device memory

    u32 allocate(u32 width, u32 height) {
        const u32 state_count = width * height;

        cudaMalloc(&state_buffer[0], state_count * sizeof(PathState));
        cudaMalloc(&state_buffer[1], state_count * sizeof(PathState));
        cudaMalloc(&counter, sizeof(StateCounter));
    }
};
#endif

// Basically, you double buffer the path states
// Another approach where you don't need to double buffer is, where you keep
// one buffer of path states and then have another array that indexes into
// the path state array. The benefit of this is that you use less memory
// and you don't have to immediately write the result away when a path
// finishes. The downside is that memory access is no longer linear.
// At this point I am not sure which one would be better.

struct StateCounter {
    u32 extend_jobs;
    u32 shade_jobs;
};

struct alignas(16) PathState {
    vec3 ray_pos; f32 t; // 16 bytes
    vec3 ray_dir; u32 triangle_index; // 16 bytes

    vec3 throughput; // 12 bytes
    RngXor32 rng; // 4 bytes

    BVHTriangleIntersection isect; // 16 bytes

    u32 pixel_index; // 4 bytes
    f32 u, v; // 8 bytes
}; // total 76 bytes

__global__ void reset(PathState* states0, PathState* states1, StateCounter* counter, u32 state_count) {
}

__global__ void generate_primary_rays(StateCounter* counter, PathState* states, 
                                      PointCamera camera, u32 width, u32 height) 
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

    states[id].rng = rng;
    states[id].throughput = vec3(1.0f, 1.0f, 1.0f);
    states[id].ray = primary_ray;
    states[id].pixel_index = id;

    atomicAdd(&counter->extend_jobs, 1);
}

__global__ void extend_rays(StateCounter* counter, PathState* states,
                            BVHNode const * bvh, RenderTriangle const * triangles)
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= ray_count) {
        return;
    }

    const BVHTriangleIntersection isect = bvh_intersect_triangles(bvh, triangles, rays[id].ray);

    if (!isect.hit()) {
        return;
    }

    states[id].pixel_index = rays[id].pixel_index;
    states[id].isect = isect;

    atomicAdd(&counters.shade_jobs, 1);
}

__global__ void shade(StateCounter* counter, PathState* states, PathState* next_states,
                      RenderTriangle const * triangles, vec4* buffer) 
{
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= shade_jobs) {
        return;
    }

    const RngXor32& rng = states[id].rng;
    const BVHTriangleIntersection& isect = states[id].isect;
    const RenderTriangle& tri = triangles[isect.id];

    if (!isect.hit()) {
        return;
    }

    switch (tri.material) {
    case MATERIAL_DIFFUSE: {
        const vec3 n = triangle_normal_lerp(tri.n0, tri.n1, tri.n2, isect.u, isect.v);
        const vec3 p = ray.pos + isect.t * ray.dir;

        const vec3 scatter_sample = sample_cosine_weighted_hemisphere(rng.random_f32(), rng.random_f32());

        vec3 t, b;
        build_orthonormal_basis(n, &t, &b);
        const vec3 scatter_direction = to_world_space(scatter_sample, n, t, b);

        const vec3 brdf = tri.color();

        states[id].throughput *= brdf;

        const u32 nextid = atomicAdd(&counter.extend_jobs, 1);
        const Ray extend_ray = Ray(p + scatter_direction * 0.0001f, scatter_direction);
        states[id].ray_pos = extend_ray.pos;
        states[id].ray_dir = extend_ray.dir;
        next_states[nextid] = states[id];
    } break;
    case MATERIAL_EMISSIVE: {
        buffer[states[id].pixel_index] = states[id].throughput * tri.light_intensity * tri.color();
    } break;
    }
}

void draw_frame() {
    PathState* states0, *states1;
    generate_prim_rays(counter, states0, camera, width, height);

    extend_rays(counter, states0, bvh, triangles);

    StateCounter counter;

    for (u32 i = 0; i < 4; i++) {
        cudaMemcpy
    }

    shade(states);
}
