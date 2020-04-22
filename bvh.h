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
    u16 bminx, bminy, bminz; // 6 bytes
    u16 bmaxx, bmaxy, bmaxz; // 6 bytes
    u32 meta;                // 4 bytes

    __device__ bool is_leaf() const { return (meta & 0xf0000000) != 0; }
    __device__ u32  count()   const { return meta >> 28; }
    __device__ u32  index()   const { return (meta & 0x0fffffff); }
    __device__ AABB bounds(CBVHData cbvh) const {
        const f32 fex = Float32(0, cbvh.ex, 0).as_f32();
        const f32 fey = Float32(0, cbvh.ey, 0).as_f32();
        const f32 fez = Float32(0, cbvh.ez, 0).as_f32();

        const Vector3 bmin = cbvh.origin + vec3(fex*bminx, fey*bminy, fez*bminz);
        const Vector3 bmax = cbvh.origin + vec3(fex*bmaxx, fey*bmaxy, fez*bmaxz);

        return { bmin, bmax, };
    }
}; // 16 bytes
       
struct CBVH {
    CBVHData data;
    std::vector<CBVHNode> nodes;
};
                                                  
#ifndef __CUDACC__
// NOTE
// This will reorder the `triangles` array
std::vector<BVHNode> build_bvh_for_triangles(RenderTriangle* triangles, u32 triangle_count, 
                                             u32* bvh_depth_out, u32* bvh_max_primitives_out);

std::vector<BVHNode> build_bvh_for_spheres(RenderSphere* spheres, u32 sphere_count,
                                           u32* bvh_depth_out, u32* bvh_max_primitives_out);
 
CBVH compress_bvh(std::vector<BVHNode> bvh);
#endif

#ifdef __CUDACC__
struct alignas(16) BVHTriangleIntersection {
    u32 id;
    f32 t, u, v;

    __device__ bool hit() const { return id != UINT32_MAX; }
};

__device__ BVHTriangleIntersection bvh_intersect_triangles(BVHNode const * bvh, 
                                                           RenderTriangle const * triangles, 
                                                           Ray ray);

__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, 
                                                   RenderTriangle const * triangles, 
                                                   Ray ray);

struct BVHSphereIntersection {
    u32 id;
    f32 t;

    __device__ bool hit() const { return id != UINT32_MAX; }
};

__device__ BVHSphereIntersection bvh_intersect_sphers(BVHNode const * bvh,
                                                      RenderSphere const * spheres,
                                                      Ray ray);

__device__ bool bvh_intersect_spheres_shadowcast(BVHNode const * bvh,
                                                 RenderSphere const * spheres,
                                                 Ray ray);

__device__ BVHTriangleIntersection cbvh_intersect_triangles(CBVHData cbvh,
                                                            CBVHNode const * cnodes,
                                                            RenderTriangle const * triangles,
                                                            Ray ray);

#endif
