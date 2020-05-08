#pragma once

struct MBVH8Traversal {
    u32 base_index;

    union {
        struct { 
            u8 node_hit_mask;
            u16 pad0;
            u8 imask;
        };

        struct {
            u32 pad1          :  8; // should be 0
            u32 triangle_mask : 24;
        };

        u32 detail;
    };

    __device__ static inline MBVH8Traversal make_node_traversal(u32 base_index, u8 hit_mask, u8 imask) {
        MBVH8Traversal traversal;
        traversal.base_index = base_index;
        traversal.node_hit_mask = hit_mask;
        traversal.imask = imask;
        return traversal;
    }

    __device__ static inline MBVH8Traversal make_triangle_traversal(u32 base_index, u32 triangle_mask) {
        MBVH8Traversal traversal;
        traversal.base_index = base_index;
        traversal.pad1 = 0;
        traversal.triangle_mask = triangle_mask;
        return traversal;
    }

    // `node_hit_mask` occupies the same space as `pad1`
    // If `node_hit_mask` is not zero, there are node hits and this is a traversal for internal nodes.
    __device__ bool is_node_group() const { return node_hit_mask != 0; }
    __device__ bool is_empty()      const { return detail == 0; }
    __device__ void set_empty() { detail = 0; }
    __device__ bool has_triangles() const { return triangle_mask != 0; }
    __device__ u32  get_next_triangle() { 
        const u32 tri = __ffs(triangle_mask); // get the least significant bit set
        detail &= ~(1 << tri); // remove the triangle from our mask
        return base_index + tri - 1;
    }
    __device__ u32 get_closest_node(Vector3 ray_dir) {
        // !!! TODO !!!
        // return the index of the closest node and remove it from this traversal
        return 0;
    }
};

struct MBVH8Intersection {
    MBVH8Traversal node_traversal;
    MBVH8Traversal triangle_traversal;
};
                  
__device__ AABB get_child_aabb_for_mbvh8node(const MBVH8Node& node, u32 index) {
    const MBVH8Node::ChildNode& c = node.child[index];

    const f32 fex = Float32(0, node.ex, 0).as_f32();
    const f32 fey = Float32(0, node.ey, 0).as_f32();
    const f32 fez = Float32(0, node.ez, 0).as_f32();

    const Vector3 bmin = node.origin + vec3(fex * c.bminx, fey * c.bminy, fez * c.bminz);
    const Vector3 bmax = node.origin + vec3(fex * c.bmaxx, fey * c.bmaxy, fez * c.bmaxz);

    return { bmin, bmax, };
}

__device__ MBVH8Intersection mbvh8node_intersect(const MBVH8Node& node, Vector3 ray_pos, Vector3 ray_inv_dir) {
    u8 hit_mask = 0;
    for (u32 i = 0; i < 8; i++) {
        f32 t; // t is never used, but we need to pass it to intersect. :| maybe change this at some point
        hit_mask |= (get_child_aabb_for_mbvh8node(node, i).intersect(ray_pos, ray_inv_dir, &t)) << i;
    }

    {
        // Maybe this part is not necessary, but at the moment I just zero out the fields in the hit_mask
        // where the nodes is empty, to make sure not to have any hits for empty nodes.
        u8 empty_mask = 0;
        for (u32 i = 0; i < 8; i++) {
            hit_mask |= (node.child[i].meta == 0) << i;
        }

        hit_mask &= ~empty_mask;
    }

    u32 triangle_mask = 0;
    for (u32 i = 0; i < 8; i++) {
        // If this child is a leaf node and we hit the AABB for that leaf then we add the triangles
        if (!(node.imask & (1 << i)) && (hit_mask & (1 << i))) {
            const u8 offset = node.child[i].meta & 0b00011111;
            triangle_mask |= ((node.child[i].meta & 0b11100000) >> 5) << offset;
        }
    }

    const MBVH8Traversal node_traversal = MBVH8Traversal::make_node_traversal(node.child_base_index, 
                                                                              hit_mask & node.imask, 
                                                                              node.imask);
    const MBVH8Traversal triangle_traversal = MBVH8Traversal::make_triangle_traversal(node.triangle_base_index, 
                                                                                      triangle_mask);

    return { node_traversal, triangle_traversal };
}

__device__ BVHTriangleIntersection mbvh8_intersect_triangles(MBVH8Node const * mbvh,
                                                             RenderTriangle const * triangles,
                                                             Ray ray)
{
    const Vector3 ray_inv_dir = ray.dir.inverse();
    auto root = mbvh8node_intersect(mbvh[0], ray.pos, ray_inv_dir); 
  
    BVHTriangleIntersection best;
    best.t = FLT_MAX;
    best.id = BVH_NO_HIT;
        
    if (root.node_traversal.is_empty()) {
        return best;
    }
 
    while (root.triangle_traversal.has_triangles()) {
        const u32 id = root.triangle_traversal.get_next_triangle();
        const auto isect = triangle_intersect(ray,
                                              triangles[id].v0,
                                              triangles[id].v1,
                                              triangles[id].v2);
        if (isect.hit && isect.t < best.t) {
            best.id = id;
            best.t  = isect.t;
            best.u  = isect.u;
            best.v  = isect.v;
        }
    }
    
    Stack<MBVH8Traversal, 32> traversal_stack;
    MBVH8Traversal traversal = root.node_traversal; // G in the paper

    while (true) {
        MBVH8Traversal triangle_traversal;

        if (traversal.is_node_group()) {
            const u32 child = traversal.get_closest_node(ray.dir);
            if (!traversal.is_empty()) {
                traversal_stack.push(traversal);
            }

            auto isect = mbvh8node_intersect(mbvh[child], ray.pos, ray_inv_dir);
            traversal = isect.node_traversal;
            triangle_traversal = isect.triangle_traversal;
        } else {
            triangle_traversal = traversal;
            traversal.set_empty();
        }

        while (triangle_traversal.has_triangles()) {
            const u32 id = triangle_traversal.get_next_triangle();
            const auto isect = triangle_intersect(ray,
                                                  triangles[id].v0,
                                                  triangles[id].v1,
                                                  triangles[id].v2);
            if (isect.hit && isect.t < best.t) {
                best.id = id;
                best.t  = isect.t;
                best.u  = isect.u;
                best.v  = isect.v;
            }
        }

        if (traversal.is_empty()) {
            if (traversal_stack.is_empty()) {
                return best;
            }

            traversal = traversal_stack.pop();
        }
    }

    return best;
}

