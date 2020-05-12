#pragma once

#include <vector>

#include "dab/dab.h"
#include "aabb.h"

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

    __host__ __device__ bool is_leaf() const { return count > 0; }
};
 
// Just some sanity checking
static_assert(offsetof(BVHNode, bmin) == 0, "");
static_assert(offsetof(BVHNode, count) == 12, "");
static_assert(offsetof(BVHNode, bmax) == 16, "");
static_assert(offsetof(BVHNode, left_first) == 28, "");
static_assert(sizeof(BVHNode) == 32, "");
        
struct alignas(16) CBVHData {
    Vector3 origin; // 12 bytes
    u8 ex, ey, ez; u8 pad; // 4 bytes
}; // 16 bytes
 
struct alignas(16) CBVHNode {
    // We use 4 bits for the count, which means we can store at most 15 primitives per leaf.

    u16 bminx, bminy, bminz; // 6 bytes
    u16 bmaxx, bmaxy, bmaxz; // 6 bytes
    u32 meta;                // 4 bytes
}; // 16 bytes
       
struct CBVH {
    CBVHData data;
    std::vector<CBVHNode> nodes;
};

struct BVHBuildResult {
    std::vector<BVHNode> bvh;
    std::vector<u32>     prim_order;
};

struct BVHObject {
    u32 valid;
    std::vector<BVHNode> bvh;
    std::vector<Vector4> triangles;
};

// !!! NOTE !!!
// After the BVH is build for your primitives you still need to reorder them as
// is given in the `prim_order` vector.
BVHBuildResult build_bvh(AABB const * aabbs, u32 aabb_count, u32 partition_bin_count, u32 max_prims_in_leaf);

std::vector<BVHNode> build_bvh_for_triangles_and_reorder(std::vector<Vector4>& triangles, 
                                                         u32 partition_bin_count, 
                                                         u32 max_primitives_in_leaf);
 
CBVH compress_bvh(std::vector<BVHNode> bvh);

bool      save_bvh_object(const char* filename,
                          BVHNode const * nodes, u32 node_count,
                          Vector4 const * triangles, u32 triangle_vertex_count);
BVHObject load_bvh_object(void const * data);

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
std::vector<MBVH8Node> build_mbvh8_for_triangles_and_reorder(std::vector<Vector4> triangles);
