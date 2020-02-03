#ifndef GUARD_BVH_H
#define GUARD_BVH_H

#include <vector>
#include <algorithm>

#include "aabb.h"

struct alignas(16) BVHNode {
    // - bmin and bmax are the min and max of the bounding box of this node.
    // - count is the amount of primitives in this leaf node, if this is 0 then
    // this node is an interior node
    // - left_first is the index of the left child of this node if this is an
    // interior node, the index of the right child is left_first + 1 or if this
    // is a leaf node then left_first is the index of the first primitive
    // in the primitive array for this node.
    vec3 bmin; uint32_t count;
    vec3 bmax; uint32_t left_first;
};

static_assert(sizeof(BVHNode) == 32, "sizeof(BVHNode) should be 32 bytes");

union uvec3 {
    struct { uint32_t x, y, z; };
    float fields[3];
};

// NOTE
// This will reorder the `triangles` array
std::vector<BVHNode> build_bvh_for_triangles(vec3 const * vertices, uvec3* triangles, uint32_t triangle_count);

#endif // GUARD_BVH_H
