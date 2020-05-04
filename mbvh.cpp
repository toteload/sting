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

struct MBVH8BuildNode {
    struct ChildNode {
        enum Type { Leaf, Internal, Empty, } type;
        AABB bounds;
        union {
            u32 internal_node_index;
            u32 triangle_count;
        };
        u32 triangles[3];
    };

    AABB bounds;
    u32 child_count;
    ChildNode child[8];
};
 
constexpr f32 TRIANGLE_INTERSECT_COST = 0.01f;
constexpr f32 NODE_INTERSECT_COST     = 1.0f;
constexpr u32 MAX_PRIM_COUNT_IN_LEAF  = 3;

struct CostNode {
    enum Option : u8 { Leaf = 0x1, Internal = 0x2, Distribute = 0x3, };

    f32 cost[7];
    u8 meta[7];

    static u8 make_leaf() { return Leaf; }
    static u8 make_internal(u32 left_option, u32 right_option) { 
        return (right_option << 5) | (left_option << 2) | Internal; 
    }
    static u8 make_distribute(u32 left_option, u32 right_option) { 
        return (right_option << 5) | (left_option << 2) | Distribute; 
    }
};

struct CostDistribute {
    f32 cost;
    u32 left_option;
    u32 right_option;
};

constexpr inline f32 cost_leaf(f32 surface_area, u32 prim_count, f32 prim_intersect_cost) {
    if (prim_count <= MAX_PRIM_COUNT_IN_LEAF) {
        return (surface_area * prim_count * prim_intersect_cost);
    } else { 
        return FLT_MAX;
    }
}

inline CostDistribute cost_distribute(const std::vector<BVHNode>& nodes, 
                                      std::vector<CostNode>& cost_nodes, 
                                      u32 this_node, 
                                      u32 k) 
{
    const u32 left  = nodes[this_node].left_first;
    const u32 right = left + 1;

    f32 min_cost = FLT_MAX;
    u32 left_option, right_option;

    for (u32 i = 0; i < k; i++) {
        const f32 cost = cost_nodes[left].cost[i] + cost_nodes[right].cost[k-i-1];
        if (cost < min_cost) {
            min_cost = cost;
            left_option = i;
            right_option = k - i - 1;
        }
    }

    return { min_cost, left_option, right_option, };
}

inline CostDistribute cost_internal(f32 surface_area, 
                                    const std::vector<BVHNode>& nodes, 
                                    std::vector<CostNode>& cost_nodes, 
                                    u32 this_node, 
                                    f32 node_intersect_cost)
{
    CostDistribute cost = cost_distribute(nodes, cost_nodes, this_node, 7); 
    cost.cost += surface_area * node_intersect_cost;
    return cost;
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

    CostNode& this_cost = cost_nodes[this_node];

    const f32            leaf_cost           = cost_leaf(sa, prim_count, TRIANGLE_INTERSECT_COST);
    const CostDistribute internal_distribute = cost_internal(sa, nodes, cost_nodes, this_node, NODE_INTERSECT_COST);

    if (leaf_cost < internal_distribute.cost) {
        this_cost.cost[0] = leaf_cost;
        this_cost.meta[0] = CostNode::make_leaf();
    } else {
        this_cost.cost[0] = internal_distribute.cost;
        this_cost.meta[0] = CostNode::make_internal(internal_distribute.left_option, internal_distribute.right_option);
    }

    for (u32 i = 1; i < 7; i++) {
        const CostDistribute distribute = cost_distribute(nodes, cost_nodes, this_node, i-1);
        if (distribute.cost < this_cost.cost[i-1]) {
            this_cost.cost[i] = distribute.cost;
            this_cost.meta[i] = CostNode::make_distribute(distribute.left_option, distribute.right_option);
        } else {
            this_cost.cost[i] = this_cost.cost[i-1];
            this_cost.meta[i] = this_cost.meta[i-1];
        }
    }

    return prim_count;
}

u32 get_triangles_from_subtree(const std::vector<BVHNode>& bvh, u32 this_node, u32* triangles) {
    if (bvh[this_node].is_leaf()) {
        for (u32 i = 0; i < bvh[this_node].count; i++) {
            triangles[i] = bvh[this_node].left_first + i;
        }
        return bvh[this_node].count;
    }

    const u32 left  = bvh[this_node].left_first;
    const u32 right = left + 1;

    const u32 triangles_left  = get_triangles_from_subtree(bvh, left,  triangles);
    const u32 triangles_right = get_triangles_from_subtree(bvh, right, triangles + triangles_left);

    return triangles_left + triangles_right;
}

void build(std::vector<MBVH8BuildNode>& mbvh, 
           const std::vector<BVHNode>& bvh, 
           const std::vector<CostNode>& cost_bvh, 
           u32 mbvh_parent,
           u32 bvh_node, 
           u32 option) 
{
    const u8 meta = cost_bvh[bvh_node].meta[option];

    switch (meta & 0x3) {
    case CostNode::Leaf: {
        u32 triangles[3];
        const u32 triangle_count = get_triangles_from_subtree(bvh, bvh_node, triangles);
        if (triangle_count > 3) {
            printf("More than 3 triangles in leaf: %d\n", triangle_count);
        }

        MBVH8BuildNode& parent = mbvh[mbvh_parent];
        if (parent.child_count == 8) { 
            printf("parent already has 8 children!\n"); 
        }
        MBVH8BuildNode::ChildNode& child = parent.child[parent.child_count++];

        child.type = MBVH8BuildNode::ChildNode::Leaf;
        child.bounds = bvh[bvh_node].bounds;
        child.triangle_count = triangle_count;

        for (u32 i = 0; i < triangle_count; i++) {
            child.triangles[i] = triangles[i];
        }
    } break;
    case CostNode::Internal: {
        MBVH8BuildNode node;
        node.bounds = bvh[bvh_node].bounds;
        node.child_count = 0;
        for (u32 i = 0; i < 8; i++) {
            node.child[i].type = MBVH8BuildNode::ChildNode::Empty;
        }

        const u32 mbvh_index = mbvh.size();
        mbvh.push_back(node);

        MBVH8BuildNode& parent = mbvh[mbvh_parent];
        if (parent.child_count == 8) { 
            printf("parent already has 8 children!\n"); 
        }
        MBVH8BuildNode::ChildNode& child = parent.child[parent.child_count++];
        child.type = MBVH8BuildNode::ChildNode::Internal;
        child.bounds = bvh[bvh_node].bounds;
        child.internal_node_index = mbvh_index;

        const u32 left_option  = (meta >> 2) & 0x7;
        const u32 right_option = (meta >> 5) & 0x7;
        const u32 left  = bvh[bvh_node].left_first;
        const u32 right = left + 1;

        build(mbvh, bvh, cost_bvh, mbvh_index, left,  left_option);
        build(mbvh, bvh, cost_bvh, mbvh_index, right, right_option);
    } break;
    case CostNode::Distribute: {
        const u32 left_option  = (meta >> 2) & 0x7;
        const u32 right_option = (meta >> 5) & 0x7;
        const u32 left  = bvh[bvh_node].left_first;
        const u32 right = left + 1;

        build(mbvh, bvh, cost_bvh, mbvh_parent, left,  left_option);
        build(mbvh, bvh, cost_bvh, mbvh_parent, right, right_option);
    } break;
    }
}
     
std::vector<MBVH8Node> build_mbvh8_for_triangles(RenderTriangle* triangles, u32 triangle_count) {
    std::vector<BVHNode> bvh = build_bvh_for_triangles(triangles, triangle_count, 12, 1);
    std::vector<CostNode> cost_bvh;
    cost_bvh.resize(bvh.size());
    u32 total_prims = calculate_cost(bvh, cost_bvh, 0);

    std::vector<MBVH8BuildNode> mbvh;

    MBVH8BuildNode root;
    root.bounds = bvh[0].bounds;
    root.child_count = 0;
    for (u32 i = 0; i < 8; i++) {
        root.child[i].type = MBVH8BuildNode::ChildNode::Empty;
    }
    mbvh.push_back(root);
    
    if ((cost_bvh[0].meta[0] & 0x3) == CostNode::Internal) {
        const u32 left_option  = (cost_bvh[0].meta[0] >> 2) & 0x7;
        const u32 right_option = (cost_bvh[0].meta[0] >> 5) & 0x7;
        const u32 left  = bvh[0].left_first;
        const u32 right = left + 1;

        build(mbvh, bvh, cost_bvh, 0, left,  left_option);
        build(mbvh, bvh, cost_bvh, 0, right, right_option);
    } else {
        build(mbvh, bvh, cost_bvh, 0, 0, 0);
    }

    u32 mbvhprims = 0;
    u32 emptynodes = 0;
    for (u32 i = 0; i < mbvh.size(); i++) {
        if (mbvh[i].child_count == 0) { emptynodes++; }
        for (u32 j = 0; j < mbvh[i].child_count; j++) {
            mbvhprims += (mbvh[i].child[j].type == MBVH8BuildNode::ChildNode::Leaf) ? mbvh[i].child[j].triangle_count : 0;
        }
    }

    printf("%llu build nodes constructed, with %d prims and %d empty nodes\n", mbvh.size(), mbvhprims, emptynodes);

    return { };
}
