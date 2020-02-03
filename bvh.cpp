#include "bvh.h"

union alignas(16) BVHAABB {
    AABB bounds;
    struct {
        alignas(16) vec3 bmin; uint32_t index;
        alignas(16) vec3 bmax;
    };
};

static_assert(sizeof(BVHAABB) == sizeof(AABB), "sizes should be the same");

struct PartitionResult {
    uint32_t should_split;
    uint32_t left_count;
};

static PartitionResult partition(BVHAABB* aabbs, AABB node_bounds, uint32_t start, uint32_t end) {
    const size_t BIN_COUNT = 16;

    float lowest_cost = FLT_MAX;
    float split_coord, split_axis;

    for (uint32_t bin = 1; bin < BIN_COUNT; bin++) {
        const float offset = cast(float, bin) / BIN_COUNT;

        uint32_t left_count = 0, right_count = 0;
        AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();

        for (uint32_t axis = 0; axis < 3; axis++) {
            const float split = node_bounds.min[axis] + offset * (node_bounds.max[axis] - node_bounds.min[axis]);

            for (uint32_t i = start; i < end; i++) {
                if (aabbs[i].bounds.centroid()[axis] <= split) {
                    left_bounds.merge(aabbs[i].bounds);
                    left_count++;
                } else {
                    right_bounds.merge(aabbs[i].bounds);
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
        return { .should_split = false };
    }

    uint32_t first_right = start;

    for (uint32_t i = start; i < end; i++) {
        if (aabbs[i].bounds.centroid()[split_axis] <= split_coord) {
            std::swap(aabbs[first_right], aabbs[i]);
            first_right++;
        }
    }

    return { .should_split = true, .left_count = first_right };
}

static void build_recursive(BVHAABB* aabbs, std::vector<BVHNode>& nodes, 
                          uint32_t start, uint32_t end, uint32_t this_node, uint32_t depth = 0) 
{
    const uint32_t count = end - start;

    AABB bounds = AABB::empty();
    for (uint32_t i = start; i < end; i++) {
        bounds = bounds.merge(aabbs[i].bounds);
    }

    nodes[this_node].bmin = bounds.min;
    nodes[this_node].bmax = bounds.max;

    // TODO also stop when too deep
    if (count <= 4) {
        // This is a leaf node
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return;
    }

    const PartitionResult result = partition(aabbs, bounds, start, end);

    if (!result.should_split) {
        // This is a leaf node
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return;
    }

    // Reserve a left and right node
    const uint32_t left = nodes.size();
    nodes.push_back({});
    nodes.push_back({});

    const uint32_t mid = start + result.left_count;
    build_recursive(aabbs, nodes, start, mid, depth + 1);
    build_recursive(aabbs, nodes, mid,   end, depth + 1);

    nodes[this_node].count = 0;
    nodes[this_node].left_first = left;
}

std::vector<BVHNode> build_bvh_for_triangles(vec3 const * vertices, uvec3* triangles, uint32_t triangle_count) {
    std::vector<BVHAABB> aabbs;
    aabbs.reserve(triangle_count);

    for (uint32_t i = 0; i < triangle_count; i++) {
        const AABB bounds = AABB::for_triangle(vertices[triangles[i].x],
                                               vertices[triangles[i].y],
                                               vertices[triangles[i].z]);
        const BVHAABB aabb = { .bmin = bounds.min, .bmax = bounds.max, .index = i };
        aabbs.push_back(aabb);
    }

    std::vector<BVHNode> nodes;
    nodes.push_back({});

    AABB bounds = AABB::empty();
    for (uint32_t i = 0; i < triangle_count; i++) {
        bounds = bounds.merge(aabbs[i].bounds);
    }

    nodes[0].bmin = bounds.min;
    nodes[0].bmax = bounds.max;

    build_recursive(aabbs.data(), nodes, 0, triangle_count, 0);

    if (nodes.size() >= 2) {
        nodes[0].count = 0;
        nodes[0].left_first = 1;
    }

    // Reorder the triangles
    std::vector<uvec3> tmp(triangles, triangles + triangle_count);
    for (uint32_t i = 0; i < triangle_count; i++) {
        triangles[i] = tmp[aabbs[i].index];
    }

    return nodes;
}
