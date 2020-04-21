#pragma once

#include <vector>

#include "dab/dab.h"
#include "aabb.h"
#include "render_data.h"

struct alignas(16) BVHNode {
    union {
        AABB bounds;

        // - bmin and bmax are the min and max of the bounding box of this node.
        //
        // - count is the amount of primitives in this leaf node, if this is 0 then
        // this node is an interior node
        //
        // - left_first is the index of the left child of this node if this is an
        // interior node, the index of the right child is left_first + 1 or if this
        // is a leaf node then left_first is the index of the first primitive
        // in the primitive array for this node.

        struct {
            Vector3 bmin; u32 count;
            Vector3 bmax; u32 left_first;
        };
    };

    __device__ bool is_leaf() const { return count > 0; }
};

// Just some sanity checking
static_assert(offsetof(BVHNode, bmin) == 0, "");
static_assert(offsetof(BVHNode, count) == 12, "");
static_assert(offsetof(BVHNode, bmax) == 16, "");
static_assert(offsetof(BVHNode, left_first) == 28, "");
static_assert(sizeof(BVHNode) == 32, "");
                                                          
#ifndef __CUDACC__
// NOTE
// This will reorder the `triangles` array
std::vector<BVHNode> build_bvh_for_triangles(RenderTriangle* triangles, u32 triangle_count, 
                                             u32* bvh_depth_out, u32* bvh_max_primitives_out);
#endif

#ifdef __CUDACC__
struct alignas(16) BVHTriangleIntersection {
    u32 id; // the most significant bit will be set if there is no hit
    f32 t, u, v;

    __device__ bool hit() const { return id != UINT32_MAX; }
};

__device__ BVHTriangleIntersection bvh_intersect_triangles(BVHNode const * bvh, 
                                                           RenderTriangle const * triangles, 
                                                           Ray ray);

__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, 
                                                   RenderTriangle const * triangles, 
                                                   Ray ray);
#endif
