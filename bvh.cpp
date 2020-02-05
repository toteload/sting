#include "bvh.h"
#include "common.h"

#ifndef __CUDACC__
const uint32_t BVH_PARTITION_BIN_COUNT = 8;
const uint32_t BVH_MAX_PRIMITIVES_IN_LEAF = 8;
const uint32_t BVH_MAX_DEPTH = 16;

union alignas(16) BVHAABB {
    AABB bounds;

    struct {
        alignas(16) vec3 bmin; uint32_t index;
        alignas(16) vec3 bmax;
    };

    BVHAABB(vec3 bmin, vec3 bmax, uint32_t index) : bmin(bmin), index(index), bmax(bmax) { }
};

static_assert(sizeof(BVHAABB) == sizeof(AABB), "sizes should be the same");

struct PartitionResult {
    bool should_split;
    uint32_t left_count;
};

PartitionResult partition(BVHAABB* aabbs, AABB node_bounds, uint32_t start, uint32_t end) {
    float lowest_cost = FLT_MAX;
    float split_coord; 
	uint32_t split_axis;

    for (uint32_t bin = 1; bin < BVH_PARTITION_BIN_COUNT; bin++) {
        const float offset = cast(float, bin) / BVH_PARTITION_BIN_COUNT;

        for (uint32_t axis = 0; axis < 3; axis++) {
            const float split = node_bounds.bmin[axis] + offset * (node_bounds.bmax[axis] - node_bounds.bmin[axis]);

            uint32_t left_count = 0, right_count = 0;
            AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();

            for (uint32_t i = start; i < end; i++) {
                if (aabbs[i].bounds.centroid()[axis] <= split) {
                    left_bounds = left_bounds.merge(aabbs[i].bounds);
                    left_count++;
                } else {
                    right_bounds = right_bounds.merge(aabbs[i].bounds);
                    right_count++;
                }
            }

            const float cost = left_bounds.surface_area() * left_count + right_bounds.surface_area() * right_count;
            if (cost < lowest_cost) {
                lowest_cost = cost;
                split_coord = split;
                split_axis = axis;
            }
        }
    }

    const float current_node_cost = node_bounds.surface_area() * (end - start);

    if (current_node_cost <= lowest_cost) {
        return { false, 0 };
    }

    uint32_t first_right = start;

    for (uint32_t i = start; i < end; i++) {
        if (aabbs[i].bounds.centroid()[split_axis] <= split_coord) {
            std::swap(aabbs[first_right], aabbs[i]);
            first_right++;
        }
    }

    return { true, first_right - start };
}

void build_recursive(BVHAABB* aabbs, std::vector<BVHNode>& nodes, 
                     uint32_t start, uint32_t end, 
                     uint32_t this_node, uint32_t depth) 
{
    const uint32_t count = end - start;

    if (count <= BVH_MAX_PRIMITIVES_IN_LEAF || depth == BVH_MAX_DEPTH) {
        // This is a leaf node
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return;
    }

    const PartitionResult result = partition(aabbs, nodes[this_node].bounds, start, end);

    if (!result.should_split) {
        // It is not worth it to split so we just keep this node as a leaf
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return;
    }

    const uint32_t left = nodes.size();

    nodes[this_node].count = 0;
    nodes[this_node].left_first = left;

    // Reserve a left and right node
    nodes.push_back({});
    nodes.push_back({});

    const uint32_t mid = start + result.left_count;

    AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();
    for (uint32_t i = start; i < mid; i++) {
        left_bounds = left_bounds.merge(aabbs[i].bounds);
    }

    for (uint32_t i = mid; i < end; i++) {
        right_bounds = right_bounds.merge(aabbs[i].bounds);
    }

    nodes[left].bounds = left_bounds;
    nodes[left+1].bounds = right_bounds;

    build_recursive(aabbs, nodes, start, mid, left,     depth + 1);
    build_recursive(aabbs, nodes, mid,   end, left + 1, depth + 1);
}

std::vector<BVHNode> build_bvh_for_triangles(RenderTriangle* triangles, uint32_t triangle_count) {
    std::vector<BVHAABB> aabbs;
    aabbs.reserve(triangle_count);

    AABB root_bounds = AABB::empty();
    for (uint32_t i = 0; i < triangle_count; i++) {
        const AABB bounds = AABB::for_triangle(triangles[i].v0,
                                               triangles[i].v1,
                                               triangles[i].v2);
        aabbs.push_back(BVHAABB(bounds.bmin, bounds.bmax, i));

        root_bounds = root_bounds.merge(bounds);
    }

    std::vector<BVHNode> nodes;
    nodes.push_back({});

    nodes[0].bounds = root_bounds;

    build_recursive(aabbs.data(), nodes, 0, triangle_count, 0, 0);

    if (nodes.size() > 1) {
        nodes[0].count = 0;
        nodes[0].left_first = 1;
    }

    // Reorder the triangles
    std::vector<RenderTriangle> tmp(triangles, triangles + triangle_count);
    for (uint32_t i = 0; i < triangle_count; i++) {
        triangles[i] = tmp[aabbs[i].index];
    }

    return nodes;
}
#endif

#ifdef __CUDACC__
template<typename T, uint32_t Capacity>
struct Stack {
    T data[Capacity];
    uint32_t top;

    Stack() : top(0) { }
    __device__ Stack(std::initializer_list<T> l) : top(l.size()) {
        for (uint32_t i = 0; i < l.size(); i++) {
            data[i] = l.begin()[i];
        }
    }

    __device__ bool is_empty() const { return top == 0; }
    uint32_t capacity() const { return Capacity; }
    uint32_t size() const { return top; }
    __device__ void push(T v) { data[top++] = v; }
    __device__ T pop() { return data[--top]; }
    T peek() const { return data[top-1]; }
};

__device__ bool bvh_intersect_triangles(BVHNode const * bvh, RenderTriangle const * triangles, Ray ray, 
                                        float* t_out, uint32_t* tri_id_out, 
                                        uint32_t* aabb_isect_count_out, uint32_t* tri_isect_count_out) 
{
    // Create a stack and initialize it with the root node
    Stack<uint32_t, 32> stack({ 0 });

    const vec3 ray_inv_dir = ray.dir.inverse();

    float t_nearest = FLT_MAX;
    uint32_t tri_id = 0;

    uint32_t aabb_isect_count = 0;
    uint32_t tri_isect_count = 0;

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.count; i++) {
                float t;
                if (triangle_intersect(ray, 
                                       triangles[node.left_first+i].v0, 
                                       triangles[node.left_first+i].v1, 
                                       triangles[node.left_first+i].v2, 
                                       &t))
                {
                    if (t < t_nearest) {
                        t_nearest = t;
                        tri_id = node.left_first + i;
                    }
                }
            }

            tri_isect_count += node.count;
        } else {
            const BVHNode& left = bvh[node.left_first];
            const BVHNode& right = bvh[node.left_first + 1];

            aabb_isect_count += 1;

            float t_left, t_right;
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

    *t_out = t_nearest;
    *tri_id_out = tri_id;
    *aabb_isect_count_out = aabb_isect_count;
    *tri_isect_count_out = tri_isect_count;

    return t_nearest != FLT_MAX;
}

__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, RenderTriangle const * triangles, 
                                                   Ray ray, float max_dist) 
{
    Stack<uint32_t, 32> stack({ 0 });

    const vec3 ray_inv_dir = ray.dir.inverse();

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.count; i++) {
                float t;
                const bool hit = triangle_intersect(ray, 
                                                    triangles[node.left_first+i].v0, 
                                                    triangles[node.left_first+i].v1, 
                                                    triangles[node.left_first+i].v2, 
                                                    &t);
                if (hit && t < max_dist) {
                    return true;
                }
            }
        } else {
            const BVHNode& left = bvh[node.left_first];
            const BVHNode& right = bvh[node.left_first + 1];

            float t_left, t_right;
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
#endif
