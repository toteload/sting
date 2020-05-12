constexpr u32 BVH_NO_HIT = UINT32_MAX;

struct alignas(16) BVHTriangleIntersection {
    u32 id;
    f32 t, u, v;

    __device__ bool hit() const { return id != UINT32_MAX; }
};

__device__ BVHTriangleIntersection bvh_intersect_triangles(BVHNode const * bvh, 
                                                           Vector4 const * triangles, 
                                                           Ray ray) 
{
    // Create a stack and initialize it with the root node
    Stack<u32, 32> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    BVHTriangleIntersection best;
    best.t = FLT_MAX;
    best.id = BVH_NO_HIT;

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];

        // Syncing threads here gives a nice, little speedup
        __syncthreads();

        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count; i++) {
                const auto isect = triangle_intersect(ray, 
                                                      vec3(triangles[(node.left_first + i) * 3 + 0]), 
                                                      vec3(triangles[(node.left_first + i) * 3 + 1]), 
                                                      vec3(triangles[(node.left_first + i) * 3 + 2]));
                if (isect.hit && isect.t < best.t) {
                    best.id = node.left_first + i;
                    best.t  = isect.t;
                    best.u  = isect.u;
                    best.v  = isect.v;
                }
            }
        } else {
            f32 t_left, t_right;
            const bool hit_left  = bvh[node.left_first].bounds.intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = bvh[node.left_first + 1].bounds.intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left > t_right) {
                    stack.push(node.left_first);
                    stack.push(node.left_first + 1);
                } else {
                    stack.push(node.left_first + 1);
                    stack.push(node.left_first);
                }
            } else {
                if (hit_left)  { stack.push(node.left_first); }
                if (hit_right) { stack.push(node.left_first + 1); }
            }
        }
    }

    return best;
}

__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, 
                                                   Vector4 const * triangles, 
                                                   Ray ray) 
{
    Stack<u32, 32> stack({ 0 });

    const Vector3 ray_inv_dir = ray.dir.inverse();

    while (!stack.is_empty()) {
        const BVHNode& node = bvh[stack.pop()];
     
        // Syncing threads here gives a nice, little speedup
        __syncthreads();
                
        if (node.is_leaf()) {
            for (u32 i = 0; i < node.count; i++) {
                const auto isect = triangle_intersect(ray, 
                                                      vec3(triangles[(node.left_first + i) * 3    ]), 
                                                      vec3(triangles[(node.left_first + i) * 3 + 1]), 
                                                      vec3(triangles[(node.left_first + i) * 3 + 2]));
                if (isect.hit && isect.t < ray.tmax) {
                    return true;
                }
            }
        } else {
            const BVHNode& left = bvh[node.left_first];
            const BVHNode& right = bvh[node.left_first + 1];

            f32 t_left, t_right;
            const bool hit_left = left.bounds.intersect(ray.pos, ray_inv_dir, &t_left);
            const bool hit_right = right.bounds.intersect(ray.pos, ray_inv_dir, &t_right);

            if (hit_left && hit_right) {
                if (t_left < t_right) {
                    stack.push(node.left_first);
                    stack.push(node.left_first + 1);
                } else {
                    stack.push(node.left_first + 1);
                    stack.push(node.left_first);
                }
            } else {
                if (hit_left)  { stack.push(node.left_first); }
                if (hit_right) { stack.push(node.left_first + 1); }
            }
        }
    }

    return false;
}

__device__ inline AABB cbvh_bounds(const CBVHData& cbvh, const CBVHNode& node) {
    const f32 fex = Float32(0, cbvh.ex, 0).as_f32();
    const f32 fey = Float32(0, cbvh.ey, 0).as_f32();
    const f32 fez = Float32(0, cbvh.ez, 0).as_f32();

    const Vector3 bmin = cbvh.origin + vec3(fex*node.bminx, fey*node.bminy, fez*node.bminz);
    const Vector3 bmax = cbvh.origin + vec3(fex*node.bmaxx, fey*node.bmaxy, fez*node.bmaxz);

    return { bmin, bmax, };
}

__device__ BVHTriangleIntersection cbvh_intersect_triangles(CBVHData cbvh_data,
                                                            CBVHNode const * cbvh,
                                                            Vector4 const * triangles,
                                                            Ray ray)
{
    // Create a stack and initialize it with the root node
#define BLOCK_SIZE 64
#define STACK_SIZE 20
    __shared__ u32 stack_memory[BLOCK_SIZE * STACK_SIZE];
    u32* const stack = stack_memory + threadIdx.x * STACK_SIZE;
    u32 stack_top = 1;
    stack[0] = 0;
#undef BLOCK_SIZE
#undef STACK_SIZE

    const Vector3 ray_inv_dir = ray.dir.inverse();

    BVHTriangleIntersection best;
    best.t = FLT_MAX;
    best.id = BVH_NO_HIT;

    while (stack_top != 0) {
        const CBVHNode& node = cbvh[stack[--stack_top]];
        const u32 id = (node.meta & 0x0fffffff);

        if ((node.meta & 0xf0000000) != 0) {
            for (u32 i = 0; i < (node.meta >> 28); i++) {
                const auto isect = triangle_intersect(ray, 
                                                      vec3(triangles[(id + i) * 3    ]), 
                                                      vec3(triangles[(id + i) * 3 + 1]), 
                                                      vec3(triangles[(id + i) * 3 + 2]));
                if (isect.hit && isect.t < best.t) {
                    best.id = id + i;
                    best.t  = isect.t;
                    best.u  = isect.u;
                    best.v  = isect.v;
                }
            }
        } else {
            f32 t_left, t_right;
            const u32 hit_left  = cbvh_bounds(cbvh_data, cbvh[id  ]).intersect(ray.pos, ray_inv_dir, &t_left);
            const u32 hit_right = cbvh_bounds(cbvh_data, cbvh[id+1]).intersect(ray.pos, ray_inv_dir, &t_right);

            const u32 hit_count = cast(u32, hit_left) + cast(u32, hit_right);

            if (hit_left && hit_right) {
                const u32 right_first = cast(u32, t_left > t_right);
                stack[stack_top  ] = id + 1 - right_first;
                stack[stack_top+1] = id +     right_first;
            } else {
                stack[stack_top] = id + cast(u32, hit_right);
            }

            stack_top += hit_count;
        }
    }

    return best;
}
 
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
                                                             Vector4 const * triangles,
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
                                              vec3(triangles[id * 3    ]),
                                              vec3(triangles[id * 3 + 1]),
                                              vec3(triangles[id * 3 + 2]));
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
                                                  vec3(triangles[id * 3    ]),
                                                  vec3(triangles[id * 3 + 1]),
                                                  vec3(triangles[id * 3 + 2]));
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

