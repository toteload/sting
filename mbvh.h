#pragma once

#include "dab/dab.h"
 
// Wide BVH Node with 8 child nodes
struct alignas(16) MBVH8Node {
    struct ChildNode {
        // - empty node, meta == 0b00000000 
        //
        // - internal node, top 3 bits of meta are 0b001, the other 5 bits store
        //   the child slot index, we need only 8 different values
        //
        // - leaf node, top 2 bits are the number of triangles stored in binary
        //   encoding. The 3rd bit is 0b1. The lower 5 bits store the index
        //   relative to triangle_base_index.
        u8 meta;

        // bounding box information for this child node
        u8 bminx, bminy, bminz;
        u8 bmaxx, bmaxy, bmaxz;
    }; // 7 bytes;

    // origin, the origin of the bounding box for this node. All the child node
    // bounding boxes are relative to this point.
    Vector3 origin; // 12 bytes

    // exponent values for every dimension of our bounding box
    u8 ex, ey, ez; // 3 bytes

    // 8-bit mask to denote which of the children are internal nodes.
    u8 imask; // 1 bytes

    //
    u32 child_base_index; // 4 bytes

    //
    u32 triangle_base_index; // 4 bytes

    //
    ChildNode nodes[8]; // 56 bytes


    AABB get_child_aabb(u32 index) {
        const ChildNode& c = nodes[index];

        const Float32 sx(0, ex, 0); 
        const Float32 sy(0, ey, 0); 
        const Float32 sz(0, ez, 0); 

        const Vector3 bmin = origin + vec3(sx * c.bminx, sy * c.bminy, sz * c.bminz);
        const Vector3 bmax = origin + vec3(sx * c.bmaxx, sy * c.bmaxy, sz * c.bmaxz);

        return { bmin, bmax };
    }
}; // 80 bytes

static_assert(sizeof(MBVH8Node) == 80, "");
        
std::vector<MBVH8Node> build_mbvh8_for_triangles(RenderTriangle* triangles, u32 triangle_count);

#ifdef __CUDACC__

__device__ BVHTriangleIntersection mbvh8_intersect_triangles(MBVH8Node const * bvh,
                                                             RenderTriangle const * triangles,
                                                             Ray ray);

#endif
