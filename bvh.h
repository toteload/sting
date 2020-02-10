#ifndef GUARD_BVH_H
#define GUARD_BVH_H

#include <vector>

#include "aabb.h"

struct alignas(16) BVHNode {
    union {
        AABB bounds;

        // - bmin and bmax are the min and max of the bounding box of this node.
        // - count is the amount of primitives in this leaf node, if this is 0 then
        // this node is an interior node
        // - left_first is the index of the left child of this node if this is an
        // interior node, the index of the right child is left_first + 1 or if this
        // is a leaf node then left_first is the index of the first primitive
        // in the primitive array for this node.
        struct {
            vec3 bmin; uint32_t count;
            vec3 bmax; uint32_t left_first;
        };
    };

    BVHNode() { }

    __device__ bool is_leaf() const { return count > 0; }
    bool intersect(vec3 ray_pos, vec3 ray_inv_dir) const;
};

// Just some sanity checking
static_assert(offsetof(BVHNode, bmin) == 0, "");
static_assert(offsetof(BVHNode, count) == 12, "");
static_assert(offsetof(BVHNode, bmax) == 16, "");
static_assert(offsetof(BVHNode, left_first) == 28, "");
static_assert(sizeof(BVHNode) == 32, "");

enum RenderMaterial : u32 {
    MATERIAL_DIFFUSE = 1,
    MATERIAL_MIRROR = 2,
    MATERIAL_EMISSIVE = 3,
};

// NOTE
// This triangle format has not yet been optimized for size.
// It is the way it is for the sake of simplicity
struct alignas(16) RenderTriangle {
    vec3 v0; uint32_t material;
    vec3 v1; uint32_t pad1;
    vec3 v2; uint32_t pad2;
    vec3 n0; uint32_t pad3;
    vec3 n1; uint32_t pad4;
    vec3 n2; uint32_t pad5;
    vec3 color; float light_intensity;
    vec3 face_normal; u32 pad6;

    RenderTriangle(vec3 v0, vec3 v1, vec3 v2) :
        v0(v0), v1(v1), v2(v2), material(MATERIAL_DIFFUSE), color(1.0f), light_intensity(1.0f)
    {
        n0 = n1 = n2 = face_normal = triangle_normal(v2, v1, v0);
    }

    RenderTriangle(vec3 v0, vec3 v1, vec3 v2, vec3 n0, vec3 n1, vec3 n2) : 
        v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2), material(MATERIAL_DIFFUSE), color(1.0f), light_intensity(1.0f)
    { 
        face_normal = triangle_normal(v2, v1, v0);
    }
};

#ifndef __CUDACC__

// NOTE
// This will reorder the `triangles` array
std::vector<BVHNode> build_bvh_for_triangles(RenderTriangle* triangles, uint32_t triangle_count);
#endif

#ifdef __CUDACC__
__device__ bool bvh_intersect_triangles(BVHNode const * bvh, RenderTriangle const * triangles, 
                                        Ray ray, float* t_out, uint32_t* tri_id_out, 
                                        uint32_t*, uint32_t*);
__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, RenderTriangle const * triangles, 
                                                   Ray ray, float max_dist);
#endif

#endif // GUARD_BVH_H
