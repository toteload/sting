#include "mbvh.h"

struct MBVH8Traversal {
    u32 base_index;
    union {
        struct { 
            u8 node_hit_mask;
            u16 pad0;
            u8 imask;
        };
        struct {
            u32 pad1              :  8;
            u32 triangle_hit_mask : 24;
        };
    };

    static inline MBVH8Traversal make_node_traversal(u32 base_index, u8 hit_mask, u8 imask) {
        MBVH8Traversal traversal;
        traversal.base_index = base_index;
        traversal.node_hit_mask = hit_mask;
        traversal.imask = imask;
        return traversal;
    }

    static inline MBVH8Traversal make_triangle_traversal(u32 base_index, u32 hit_mask) {
        MBVH8Traversal traversal;
        traversal.base_index = base_index;
        traversal.pad1 = 0;
        traversal.triangle_hit_mask = hit_mask;
        return traversal;
    }

    // `node_hit_mask` occupies the same space as `pad1`
    // If `node_hit_mask` is not zero, there are node hits and this is a traversal for internal nodes.

    __device__ bool is_node_group() const { return node_hit_mask != 0; }
};

constexpr u32 NODE_OPTION_LEAF       = 0b0000; 
constexpr u32 NODE_OPTION_INTERNAL   = 0b0001;
constexpr u32 NODE_OPTION_DISTRIBUTE = 0b0010;

struct CostNode {
    f32 cost[7]; u32 flags; // we have 4 bits per option to save our option
};

constexpr f32 TRIANGLE_INTERSECT_COST = 1.0f;
constexpr f32 NODE_INTERSECT_COST     = 0.25f;
constexpr u32 MAX_PRIM_COUNT_IN_LEAF  = 3;

constexpr inline f32 cost_leaf(f32 surface_area, u32 prim_count, f32 prim_intersect_cost) {
    return (prim_count <= MAX_PRIM_COUNT_IN_LEAF) ? (surface_area * prim_count * prim_intersect_cost) : FLT_MAX;
}

inline f32 cost_distribute(const std::vector<BVHNode>& nodes, std::vector<CostNode>& cost_nodes, u32 this_node, u32 k) {
    const u32 left  = nodes[this_node].left_first;
    const u32 right = left + 1;

    f32 min_cost = FLT_MAX;

    for (u32 i = 0; i < k; i++) {
        const f32 cost = cost_nodes[left].cost[i] + cost_nodes[right].cost[k-i];
        min_cost = min(min_cost, cost);
    }

    return min_cost;
}

inline f32 cost_internal(f32 surface_area, 
                         const std::vector<BVHNode>& nodes, 
                         std::vector<CostNode>& cost_nodes, 
                         u32 this_node, 
                         f32 node_intersect_cost)
{
    return cost_distribute(nodes, cost_nodes, this_node, 7) + surface_area * node_intersect_cost;
}

u32 calculate_cost(const std::vector<BVHNode>& nodes, std::vector<CostNode>& cost_nodes, u32 this_node) {
    const f32 sa = nodes[this_node].bounds.surface_area();

    if (nodes[this_node].is_leaf()) {
         const f32 leaf_cost = cost_leaf(sa, 1, TRIANGLE_INTERSECT_COST);
         for (u32 i = 0; i < 7; i++) {
             cost_nodes[this_node].cost[i] = leaf_cost;
         }

         return 1;
    }

    const u32 prim_count_left  = calculate_cost(nodes, cost_nodes, nodes[this_node].left_first);
    const u32 prim_count_right = calculate_cost(nodes, cost_nodes, nodes[this_node].left_first + 1);

    const u32 prim_count = prim_count_left + prim_count_right;

#if 0
    CostNode& this_cost = cost_nodes[this_node];
    f32 min_cost;
    u32 flags = 0;

    const f32 leaf_cost = cost_leaf(sa, prim_count, TRIANGLE_INTERSECT_COST);
    min_cost = leaf_cost;
    flags = NODE_OPTION_LEAF;

    const f32 interal_cost = cost_internal(sa, nodes, cost_nodes, this_node, NODE_INTERSECT_COST);
    if (internal_cost < min_cost) {
        min_cost = internal_cost;
        flags = NODE_OPTION_INTERNAL;
    }

    this_cost.cost[0] = min(cost_leaf(sa, prim_count, TRIANGLE_INTERSECT_COST),
                            cost_internal(sa, nodes, cost_nodes, this_node, NODE_INTERSECT_COST));
    for (u32 i = 1; i < 7; i++) {
        this_cost.cost[i] = min(cost_distribute(nodes, cost_nodes, this_node, i), this_cost.cost[i-1]);
    }
#endif

    return prim_count;
}

u32 count_prims(const std::vector<BVHNode>& nodes, u32 this_node) {
    if (nodes[this_node].is_leaf()) {
        return nodes[this_node].count;
    } else {
        const u32 left  = nodes[this_node].left_first;
        const u32 right = left + 1;
        return count_prims(nodes, left) + count_prims(nodes, right);
    }
}

std::vector<MBVH8Node> build_mbvh8_for_triangles(RenderTriangle* triangles, u32 triangle_count) {
    u32 bvh_depth;
    std::vector<BVHNode> bvh = build_bvh_for_triangles(triangles, triangle_count, 12, 1, &bvh_depth);
    std::vector<CostNode> cost_bvh;
    cost_bvh.resize(bvh.size());
    u32 total_prims = calculate_cost(bvh, cost_bvh, 0);
    printf("total prims %d\n", total_prims);
    printf("total cost %f\n", cost_bvh[0].cost[0]);

    return { };
}
 
__device__ BVHTriangleIntersection mbvh8_intersect_triangles(MBVH8Node const * bvh,
                                                             RenderTriangle const * triangles,
                                                             Ray ray)
{
#if 0
    // TODO set the stack size to some better value that is not hard coded like this.
    Stack<MBVH8Traversal, 16> traversal_stack;

     const Vector3 ray_inv_dir = ray.dir.inverse();

     MBVH8Traversal 

     while (!stack.empty()) {

     }
#endif
}
 
#if 0
// This type of traversal allows for 4 triangles per leaf instead of 3
struct MBVH8Traversal {
    u32 base_index : 31;
    u32 is_triangle_traversal : 1;
    union {
        struct {
            u16 pad0;
            u8  node_hits;
            u8  imask;
        };

        u32 triangle_hits;
    };
};
#endif
                                                                    
