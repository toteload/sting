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
    ChildNode nodes[8]; // 56 bytes

    AABB get_child_aabb(u32 index) {
        const ChildNode& c = nodes[index];

        const f32 fex = Float32(0, ex, 0).as_f32();
        const f32 fey = Float32(0, ey, 0).as_f32();
        const f32 fez = Float32(0, ez, 0).as_f32();

        const Vector3 bmin = origin + vec3(fex * c.bminx, fey * c.bminy, fez * c.bminz);
        const Vector3 bmax = origin + vec3(fex * c.bmaxx, fey * c.bmaxy, fez * c.bmaxz);

        return { bmin, bmax, };
    }

#if 0
    struct MBVH8Intersection {
        MBVH8Traversal node_traversal;
        MBVH8Traversal triangle_traversal;
    };

    MBVH8Intersection intersect(Vector3 ray_pos, Vector3 ray_inv_dir, Vector3 ray_dir) {
        u8 hit_mask = 0;
        for (u32 i = 0; i < 8; i++) {
            f32 t; // never used, but we need to pass it to intersect. :| maybe change this at some point
            hit_mask |= get_child_aabb(i).intersect(ray_pos, ray_inv_dir, &t);
        }

        {
            // Maybe this part is not necessary, but at the moment I just zero out the fields in the hit_mask
            // where the nodes is empty, to make sure not to have any hits for empty nodes.
            u8 empty_mask = 0;
            for (u32 i = 0; i < 8; i++) {
                hit_mask |= (nodes[i].meta == 0) << i;
            }

            hit_mask &= ~empty_mask;
        }

        u32 triangle_hits = 0;


        const MBVH8Traversal node_traversal = MBVH8Traversal::make_node_traversal(child_base_index, hit_mask & imask, imask);
        const MBVH8Traversal triangle_traversal = MBVH8Traversal::make_triangle_traversal(triangle_base_index,
    }
#endif
}; // 80 bytes

static_assert(sizeof(MBVH8Node) == 80, "");
        
std::vector<MBVH8Node> build_mbvh8_for_triangles(RenderTriangle* triangles, u32 triangle_count);

__device__ BVHTriangleIntersection mbvh8_intersect_triangles(MBVH8Node const * bvh,
                                                             RenderTriangle const * triangles,
                                                             Ray ray);

