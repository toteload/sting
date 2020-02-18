#define PRIME0 100030001
#define PRIME1 396191693

struct PathStates {
    Ray* ray;
};

__global__ void generate_primary_rays(Ray* rays, PointCamera camera, u32 width, u32 height) {
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const u32 id = y * width + x;
    const u32 seed = (id + framenum * PRIME0) * PRIME1;
    RngXor32 rng(seed);

    // nx and ny are in range (-1.0f, 1.0f)
    const f32 nx = (2.0f * cast(f32, x) + rng.random_f32()) / width  - 1.0f;
    const f32 ny = (2.0f * cast(f32, y) + rng.random_f32()) / height - 1.0f;

    const Ray primary_ray = camera.create_ray(nx, ny);

    rays[id] = primary_ray;
}

struct PathState {
    Ray      ray;
    RngXor32 rng;
};

__global__ void extend_rays(Ray* rays, BVHNode const * bvh, RenderTriangle const * triangles, u32 width, u32 height) {
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const u32 id = y * width + x;
    const BVHTriangleIntersection isect = bvh_intersect_triangles(bvh, triangles, rays[id]);
}

__global__ void shade() {

}



