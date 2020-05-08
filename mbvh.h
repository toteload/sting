#pragma once

#include "dab/dab.h"

// Wide BVH Node with 8 child nodes
struct alignas(16) MBVH8Node {
    struct ChildNode {
        // - empty node, meta == 0b00000000 
        //
        // - internal node, top 3 bits of meta are 0b001, the other 5 bits store
        //   the child slot index + 24, so it will be in range [24..31]
        //
        // - leaf node, top 3 bits are the number of triangles stored in unary
        //   encoding. The lower 5 bits store the index
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
    ChildNode child[8]; // 56 bytes
}; // 80 bytes

static_assert(sizeof(MBVH8Node) == 80, "");

struct MBVHBuildResult {
    std::vector<MBVH8Node> mbvh;
    std::vector<u32>       prim_order;
};

MBVHBuildResult        build_mbvh8(AABB const * aabbs, u32 aabb_count);
std::vector<MBVH8Node> build_mbvh8_for_triangles_and_reorder(RenderTriangle* triangles, u32 triangle_count);
