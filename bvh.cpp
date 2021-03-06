#include "bvh.h"
#include "dab/dab.h"
#include <math.h>

#ifndef __CUDACC__
union alignas(16) BVHBuildPrimitive {
    AABB bounds;

    struct {
        alignas(16) Vector3 bmin; u32 index;
        alignas(16) Vector3 bmax;
    };

    BVHBuildPrimitive() { }
    BVHBuildPrimitive(const AABB& bounds, u32 index) : bmin(bounds.bmin), index(index), bmax(bounds.bmax) { }
    BVHBuildPrimitive(const Vector3& bmin, const Vector3& bmax, u32 index) : bmin(bmin), index(index), bmax(bmax) { }
};

static_assert(sizeof(BVHBuildPrimitive) == sizeof(AABB), "sizes should be the same");

struct PartitionResult {
    bool should_split;
    u32 left_count;
};

PartitionResult partition(BVHBuildPrimitive* aabbs, AABB node_bounds, u32 start, u32 end, u32 bin_count) {
    f32 lowest_cost = FLT_MAX;
    f32 split_coord; 
	u32 split_axis;

    for (u32 bin = 1; bin < bin_count; bin++) {
        const f32 offset = cast(f32, bin) / bin_count;

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

            // Need to do this in the case that all primitives are on one side, which causes the AABB of
            // the other side to be empty and return a surface area = inf. inf * 0.0f = nan, so we cannot
            // just multiply.
			const f32 cost = ((left_count)  ? (left_count  * left_bounds.surface_area())  : 0.0f) +
							 ((right_count) ? (right_count * right_bounds.surface_area()) : 0.0f);            

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

    return { true , first_right - start };
}

u32 build_recursive(BVHBuildPrimitive* aabbs, std::vector<BVHNode>& nodes, 
                    u32 start, u32 end, 
                    u32 this_node, u32 depth,
                    u32 partition_bin_count, u32 max_prims_in_leaf) 
{
    const u32 count = end - start;

    if (count <= max_prims_in_leaf) {
        // This is a leaf node
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return depth;
    }

    const PartitionResult result = partition(aabbs, nodes[this_node].bounds, start, end, partition_bin_count);

    u32 mid;
    if (!result.should_split) {
        // The partitioning says we should not split since it would increase cost, but we still have more
        // primitives than we would like to have for this node to be a leaf, so we split anyway.
        // The splitting we do evenly.
        mid = start + count / 2;
    } else {
        mid = start + result.left_count;
    }

    const u32 left = nodes.size();

    nodes[this_node].count = 0;
    nodes[this_node].left_first = left;

    // Reserve a left and right node
    nodes.push_back({});
    nodes.push_back({});

    AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();
    for (u32 i = start; i < mid; i++) {
        left_bounds = left_bounds.merge(aabbs[i].bounds);
    }

    for (u32 i = mid; i < end; i++) {
        right_bounds = right_bounds.merge(aabbs[i].bounds);
    }

    nodes[left  ].bounds = left_bounds;
    nodes[left+1].bounds = right_bounds;

    const u32 depth_left = build_recursive(aabbs, nodes, 
                                           start, mid, 
                                           left, depth + 1, 
                                           partition_bin_count, max_prims_in_leaf);
    const u32 depth_right = build_recursive(aabbs, nodes, 
                                            mid, end, 
                                            left + 1, depth + 1, 
                                            partition_bin_count, max_prims_in_leaf);

    return max(depth_left, depth_right);
}

BVHBuildResult build_bvh(AABB const * aabbs, u32 aabb_count, u32 partition_bin_count, u32 max_prims_in_leaf) {
    std::vector<BVHBuildPrimitive> build_prims(aabb_count);
    AABB root_bounds = AABB::empty();
    for (u32 i = 0; i < aabb_count; i++) {
        build_prims[i] = BVHBuildPrimitive(aabbs[i], i);
        root_bounds = root_bounds.merge(aabbs[i]);
    }

    std::vector<BVHNode> nodes;
    nodes.push_back({});
    nodes[0].bounds = root_bounds;

    const u32 bvh_depth = build_recursive(build_prims.data(), nodes,
                                          0, aabb_count,
                                          0, 0,
                                          partition_bin_count, max_prims_in_leaf);

    if (nodes.size() > 1) {
        nodes[0].count = 0;
        nodes[0].left_first = 1;
    }

    std::vector<u32> prim_order(aabb_count);
    for (u32 i = 0; i < aabb_count; i++) {
        prim_order[i] = build_prims[i].index;
    }

    return { .depth      = bvh_depth,
             .bvh        = nodes,
             .prim_order = prim_order, };
}

std::vector<BVHNode> build_bvh_for_triangles_and_reorder(RenderTriangle* triangles, u32 triangle_count,
                                                         u32 partition_bin_count, u32 max_prims_in_leaf) 
{
    std::vector<AABB> aabbs(triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        aabbs[i] = AABB::for_triangle(triangles[i].v0,
                                      triangles[i].v1,
                                      triangles[i].v2);
    }

    BVHBuildResult result = build_bvh(aabbs.data(), triangle_count, partition_bin_count, max_prims_in_leaf);

    // Reorder the triangles
    std::vector<RenderTriangle> tmp(triangles, triangles + triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        triangles[i] = tmp[result.prim_order[i]];
    }
    
    return result.bvh;
}

CBVH compress_bvh(std::vector<BVHNode> bvh) {
    const Vector3 origin = bvh[0].bounds.bmin;

    const u8 ex = cast(u8, 127.0f + ceil(log2((bvh[0].bounds.bmax.x - bvh[0].bounds.bmin.x) / 65535.0f)));
    const u8 ey = cast(u8, 127.0f + ceil(log2((bvh[0].bounds.bmax.y - bvh[0].bounds.bmin.y) / 65535.0f)));
    const u8 ez = cast(u8, 127.0f + ceil(log2((bvh[0].bounds.bmax.z - bvh[0].bounds.bmin.z) / 65535.0f)));

    const f32 fex = Float32(0, ex, 0).as_f32();
    const f32 fey = Float32(0, ey, 0).as_f32();
    const f32 fez = Float32(0, ez, 0).as_f32();

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
    Stack<u32, 32> stack({ 0 });

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
    Stack<u32, 32> stack({ 0 });

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

__device__ BVHTriangleIntersection cbvh_intersect_triangles(CBVHData cbvh_data,
                                                            CBVHNode const * cbvh,
                                                            RenderTriangle const * triangles,
                                                            Ray ray)
{
    // Create a stack and initialize it with the root node
    Stack<u32, 32> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    BVHTriangleIntersection best;
    best.t = FLT_MAX;
    best.id = BVH_NO_HIT;

    while (!stack.is_empty()) {
        const CBVHNode& node = cbvh[stack.pop()];
        const u32 id = node.index();

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
            const u32 hit_left  = cbvh[id]  .bounds(cbvh_data).intersect(ray.pos, ray_inv_dir, &t_left);
            const u32 hit_right = cbvh[id+1].bounds(cbvh_data).intersect(ray.pos, ray_inv_dir, &t_right);

            const u32 hit_count = cast(u32, hit_left) + cast(u32, hit_right);

            if (hit_left && hit_right) {
                const u32 right_first = cast(u32, t_left > t_right);
                stack.data[stack.top  ] = id + 1 - right_first;
                stack.data[stack.top+1] = id +     right_first;
            } else {
                stack.data[stack.top] = id + cast(u32, hit_right);
            }

            stack.top += hit_count;
        }
    }

    return best;
}
                                                                    
#endif
