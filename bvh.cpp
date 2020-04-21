#include "bvh.h"
#include "dab/dab.h"
#include <math.h>

constexpr u32 BVH_PARTITION_BIN_COUNT = 8;
constexpr u32 BVH_MAX_PRIMITIVES_IN_LEAF = 8;
constexpr u32 BVH_MAX_DEPTH = 64;

#ifndef __CUDACC__
union alignas(16) BVHAABB {
    AABB bounds;

    struct {
        alignas(16) Vector3 bmin; u32 index;
        alignas(16) Vector3 bmax;
    };

    BVHAABB(Vector3 bmin, Vector3 bmax, u32 index) : bmin(bmin), index(index), bmax(bmax) { }
};

static_assert(sizeof(BVHAABB) == sizeof(AABB), "sizes should be the same");

struct PartitionResult {
    bool should_split;
    u32 left_count;
};

PartitionResult partition(BVHAABB* aabbs, AABB node_bounds, u32 start, u32 end) {
    f32 lowest_cost = FLT_MAX;
    f32 split_coord; 
	u32 split_axis;

    for (u32 bin = 1; bin < BVH_PARTITION_BIN_COUNT; bin++) {
        const f32 offset = cast(f32, bin) / BVH_PARTITION_BIN_COUNT;

        for (u32 axis = 0; axis < 3; axis++) {
            const f32 split = node_bounds.bmin[axis] + offset * (node_bounds.bmax[axis] - node_bounds.bmin[axis]);

            u32 left_count = 0, right_count = 0;
            AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();

            for (u32 i = start; i < end; i++) {
                if (aabbs[i].bounds.centroid()[axis] <= split) {
                    left_bounds = left_bounds.merge(aabbs[i].bounds);
                    left_count++;
                } else {
                    right_bounds = right_bounds.merge(aabbs[i].bounds);
                    right_count++;
                }
            }

            const f32 cost = left_bounds.surface_area() * left_count + right_bounds.surface_area() * right_count;
            if (cost < lowest_cost) {
                lowest_cost = cost;
                split_coord = split;
                split_axis = axis;
            }
        }
    }

    const f32 current_node_cost = node_bounds.surface_area() * (end - start);

    if (current_node_cost <= lowest_cost) {
        return { false, 0 };
    }

    u32 first_right = start;

    for (u32 i = start; i < end; i++) {
        if (aabbs[i].bounds.centroid()[split_axis] <= split_coord) {
            std::swap(aabbs[first_right], aabbs[i]);
            first_right++;
        }
    }

    return { true, first_right - start };
}

void build_recursive(BVHAABB* aabbs, std::vector<BVHNode>& nodes, 
                     u32 start, u32 end, 
                     u32 this_node, u32 depth,
                     u32* bvh_max_depth_out, u32* bvh_max_primitives_out) 
{
    const u32 count = end - start;

    *bvh_max_depth_out = max(*bvh_max_depth_out, depth);

    if (count <= BVH_MAX_PRIMITIVES_IN_LEAF || depth == BVH_MAX_DEPTH) {
        // This is a leaf node
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;

        *bvh_max_primitives_out = max(*bvh_max_primitives_out, count);

        return;
    }

    const PartitionResult result = partition(aabbs, nodes[this_node].bounds, start, end);

    if (!result.should_split) {
        // It is not worth it to split so we just keep this node as a leaf
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return;
    }

    const u32 left = nodes.size();

    nodes[this_node].count = 0;
    nodes[this_node].left_first = left;

    // Reserve a left and right node
    nodes.push_back({});
    nodes.push_back({});

    const u32 mid = start + result.left_count;

    AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();
    for (u32 i = start; i < mid; i++) {
        left_bounds = left_bounds.merge(aabbs[i].bounds);
    }

    for (u32 i = mid; i < end; i++) {
        right_bounds = right_bounds.merge(aabbs[i].bounds);
    }

    nodes[left].bounds = left_bounds;
    nodes[left+1].bounds = right_bounds;

    build_recursive(aabbs, nodes, start, mid, left,     depth + 1, bvh_max_depth_out, bvh_max_primitives_out);
    build_recursive(aabbs, nodes, mid,   end, left + 1, depth + 1, bvh_max_depth_out, bvh_max_primitives_out);
}

std::vector<BVHNode> build_bvh_for_triangles(RenderTriangle* triangles, u32 triangle_count,
                                             u32* bvh_depth_out, u32* bvh_max_primitives_out) 
{
    std::vector<BVHAABB> aabbs;
    aabbs.reserve(triangle_count);

    // We precalculate the AABB for every triangle. This is much faster than
    // doing it multiple times on the fly.
    AABB root_bounds = AABB::empty();
    for (u32 i = 0; i < triangle_count; i++) {
        const AABB bounds = AABB::for_triangle(triangles[i].v0,
                                               triangles[i].v1,
                                               triangles[i].v2);
        aabbs.push_back(BVHAABB(bounds.bmin, bounds.bmax, i));

        root_bounds = root_bounds.merge(bounds);
    }

    std::vector<BVHNode> nodes;
    nodes.push_back({});

    nodes[0].bounds = root_bounds;

    u32 bvh_depth = 0;
    u32 bvh_max_primitives = 0;
    build_recursive(aabbs.data(), nodes, 0, triangle_count, 0, 0, &bvh_depth, &bvh_max_primitives);

    if (nodes.size() > 1) {
        nodes[0].count = 0;
        nodes[0].left_first = 1;
    }

    // Reorder the triangles by copying all of them and then writing back in
    // the new order
    std::vector<RenderTriangle> tmp(triangles, triangles + triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        triangles[i] = tmp[aabbs[i].index];
    }

    *bvh_depth_out = bvh_depth;
    *bvh_max_primitives_out = bvh_max_primitives;

    return nodes;
}
 
#if 0
std::vector<BVHNode> build_bvh_for_spheres(RenderSphere* spheres, u32 sphere_count,
                                           u32* bvh_depth_out, u32* bvh_max_primitives_out)
{

}
#endif

CBVH compress_bvh(std::vector<BVHNode> bvh) {
    const Vector3 origin = bvh[0].bounds.bmin;

    // Maybe this can be made faster by extracting the exponent bits from the float directly
    const i8 ex = cast(u8, ceil(log2((bvh[0].bounds.bmax.x - bvh[0].bounds.bmin.x) / 65535.0f)));
    const i8 ey = cast(u8, ceil(log2((bvh[0].bounds.bmax.x - bvh[0].bounds.bmin.x) / 65535.0f)));
    const i8 ez = cast(u8, ceil(log2((bvh[0].bounds.bmax.x - bvh[0].bounds.bmin.x) / 65535.0f)));

    const f32 fex = powf(2.0f, ex);
    const f32 fey = powf(2.0f, ey);
    const f32 fez = powf(2.0f, ez);

    std::vector<CBVHNode> cbvh;
    cbvh.reserve(bvh.size());
    
    for (u32 i = 0; i < bvh.size(); i++) {
        const Vector3 bmin = bvh[i].bmin - origin;
        const Vector3 bmax = bvh[i].bmax - origin;

        const u16 bminx = cast(u16, floor(bmin.x / fex));
        const u16 bminy = cast(u16, floor(bmin.y / fey));
        const u16 bminz = cast(u16, floor(bmin.z / fez));

        const u16 bmaxx = cast(u16, ceil(bmax.x / fex));
        const u16 bmaxy = cast(u16, ceil(bmax.y / fey));
        const u16 bmaxz = cast(u16, ceil(bmax.z / fez));

        const u32 meta = (bvh[i].count << 28) | (bvh[i].left_first & 0x0fffffff);

        cbvh.push_back({ bminx, bminy, bminz, bmaxx, bmaxy, bmaxz, meta });
    }

    return { .data = { origin, ex, ey, ez, 0, }, .nodes = cbvh };
}

#endif // ifndef __CUDACC__

#ifdef __CUDACC__
constexpr u32 BVH_NO_HIT = UINT32_MAX;

__device__ BVHTriangleIntersection bvh_intersect_triangles(BVHNode const * bvh, 
                                                           RenderTriangle const * triangles, 
                                                           Ray ray) 
{
    // Create a stack and initialize it with the root node
    Stack<u32, BVH_MAX_DEPTH> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    BVHTriangleIntersection best;
    best.t = FLT_MAX;
    best.id = BVH_NO_HIT;

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];

        // Syncing threads here gives a nice, little speedup
        __syncthreads();

        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count; i++) {
                const auto isect = triangle_intersect(ray, 
                                                      triangles[node.left_first + i].v0, 
                                                      triangles[node.left_first + i].v1, 
                                                      triangles[node.left_first + i].v2);
                if (isect.hit && isect.t < best.t) {
                    best.id = node.left_first + i;
                    best.t  = isect.t;
                    best.u  = isect.u;
                    best.v  = isect.v;
                }
            }
        } else {
            f32 t_left, t_right;
            const bool hit_left  = bvh[node.left_first].bounds.intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = bvh[node.left_first + 1].bounds.intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left > t_right) {
                    stack.push(node.left_first);
                    stack.push(node.left_first + 1);
                } else {
                    stack.push(node.left_first + 1);
                    stack.push(node.left_first);
                }
            } else {
                if (hit_left)  { stack.push(node.left_first); }
                if (hit_right) { stack.push(node.left_first + 1); }
            }
        }
    }

    return best;
}

__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, RenderTriangle const * triangles, 
                                                   Ray ray) 
{
    Stack<u32, BVH_MAX_DEPTH> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];
     
        // Syncing threads here gives a nice, little speedup
        __syncthreads();
                
        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count; i++) {
                const auto isect = triangle_intersect(ray, 
                                                      triangles[node.left_first+i].v0, 
                                                      triangles[node.left_first+i].v1, 
                                                      triangles[node.left_first+i].v2);
                if (isect.hit && isect.t < ray.tmax) {
                    return true;
                }
            }
        } else {
            const BVHNode& left = bvh[node.left_first];
            const BVHNode& right = bvh[node.left_first + 1];

            f32 t_left, t_right;
            const bool hit_left = left.bounds.intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = right.bounds.intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left < t_right) {
                    stack.push(node.left_first);
                    stack.push(node.left_first + 1);
                } else {
                    stack.push(node.left_first + 1);
                    stack.push(node.left_first);
                }
            } else {
                if (hit_left)  { stack.push(node.left_first); }
                if (hit_right) { stack.push(node.left_first + 1); }
            }
        }
    }

    return false;
}











 
__device__ BVHSphereIntersection bvh_intersect_spheres(BVHNode const * bvh, 
                                                       RenderSphere const * spheres, 
                                                       Ray ray) 
{
    // Create a stack and initialize it with the root node
    Stack<u32, BVH_MAX_DEPTH> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    BVHSphereIntersection best;
    best.t  = FLT_MAX;
    best.id = BVH_NO_HIT;

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];

        // Syncing threads here gives a nice, little speedup
        __syncthreads();

        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count; i++) {
                const auto isect = sphere_intersect(ray, 
                                                    spheres[node.left_first + i].pos, 
                                                    spheres[node.left_first + i].radius);
                if (isect.hit && isect.t < best.t) {
                    best.id = node.left_first + i;
                    best.t  = isect.t;
                }
            }
        } else {
            f32 t_left, t_right;
            const bool hit_left  = bvh[node.left_first].bounds.intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = bvh[node.left_first + 1].bounds.intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left > t_right) {
                    stack.push(node.left_first);
                    stack.push(node.left_first + 1);
                } else {
                    stack.push(node.left_first + 1);
                    stack.push(node.left_first);
                }
            } else {
                if (hit_left)  { stack.push(node.left_first); }
                if (hit_right) { stack.push(node.left_first + 1); }
            }
        }
    }

    return best;
}

__device__ bool bvh_intersect_spheres_shadowcast(BVHNode const * bvh, RenderSphere const * spheres, 
                                                 Ray ray) 
{
    Stack<u32, BVH_MAX_DEPTH> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];
     
        // Syncing threads here gives a nice, little speedup
        __syncthreads();
                
        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count; i++) {
                const auto isect = sphere_intersect(ray, 
                                                    spheres[node.left_first+i].pos, 
                                                    spheres[node.left_first+i].radius); 
                if (isect.hit && isect.t < ray.tmax) {
                    return true;
                }
            }
        } else {
            const BVHNode& left = bvh[node.left_first];
            const BVHNode& right = bvh[node.left_first + 1];

            f32 t_left, t_right;
            const bool hit_left = left.bounds.intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = right.bounds.intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left < t_right) {
                    stack.push(node.left_first);
                    stack.push(node.left_first + 1);
                } else {
                    stack.push(node.left_first + 1);
                    stack.push(node.left_first);
                }
            } else {
                if (hit_left)  { stack.push(node.left_first); }
                if (hit_right) { stack.push(node.left_first + 1); }
            }
        }
    }

    return false;
}
 
__device__ BVHTriangleIntersection cbvh_intersect_triangles(CBVHData cbvh_data,
                                                            CBVHNode const * cbvh,
                                                            RenderTriangle const * triangles,
                                                            Ray ray)
{
    // Create a stack and initialize it with the root node
    Stack<u32, BVH_MAX_DEPTH*2> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    BVHTriangleIntersection best;
    best.t = FLT_MAX;
    best.id = BVH_NO_HIT;

    while (!stack.is_empty()) {
        const CBVHNode& node = cbvh[stack.pop()];
        const u32 id = node.index();

        // Syncing threads here gives a nice, little speedup
        __syncthreads();

        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count(); i++) {
                const auto isect = triangle_intersect(ray, 
                                                      triangles[id + i].v0, 
                                                      triangles[id + i].v1, 
                                                      triangles[id + i].v2);
                if (isect.hit && isect.t < best.t) {
                    best.id = id + i;
                    best.t  = isect.t;
                    best.u  = isect.u;
                    best.v  = isect.v;
                }
            }
        } else {
            f32 t_left, t_right;
            const bool hit_left  = cbvh[id]  .bounds(cbvh_data).intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = cbvh[id+1].bounds(cbvh_data).intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left > t_right) {
                    stack.push(id  );
                    stack.push(id+1);
                } else {
                    stack.push(id+1);
                    stack.push(id  );
                }
            } else {
                if (hit_left)  { stack.push(id  ); }
                if (hit_right) { stack.push(id+1); }
            }
        }
    }

    return best;}
                                                                    
#endif
