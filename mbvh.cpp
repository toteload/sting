
struct MBVH8Traversal {
    u32 base_index;
    union {
        struct { 
            u8 node_hit_mask;
            u16 pad0;
            u8 imask;
        };
        u32 pad1              :  8;
        u32 triangle_hit_mask : 24;
    };
};
 
__device__ BVHTriangleIntersection mbvh8_intersect_triangles(MBVH8Node const * bvh,
                                                             RenderTriangle const * triangles,
                                                             Ray ray)
{
     const Vector3 ray_inv_dir = ray.dir.inverse();

     while (!stack.empty()) {

     }
}
                                                                     
