#include "mbvh.h"

struct MBVHBuildNode {
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
 
constexpr f32 TRIANGLE_INTERSECT_COST = 0.3f;
constexpr f32 NODE_INTERSECT_COST     = 1.0f;
constexpr u32 MAX_PRIM_COUNT_IN_LEAF  = 3;

struct CostNode {
    enum Option : u8 { Leaf = 0x1, Internal = 0x2, Distribute = 0x3, };

    f32 cost[7];
    u8 meta[7];

    static u8 make_leaf() { return Leaf; }
    static u8 make_internal(u32 left_option, u32 right_option) { 
        assert(left_option <= 7 && right_option <= 7);
        return (right_option << 5) | (left_option << 2) | Internal; 
    }
    static u8 make_distribute(u32 left_option, u32 right_option) { 
        assert(left_option <= 7 && right_option <= 7);
        return (right_option << 5) | (left_option << 2) | Distribute; 
    }
};

struct CostDistribute {
    f32 cost;
    u32 left_option;
    u32 right_option;
};

inline f32 cost_leaf(f32 surface_area, u32 prim_count, f32 prim_intersect_cost) {
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
             cost_nodes[this_node].meta[i] = CostNode::make_leaf();
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

// Gathers all the triangles from in a given subtree of which the root node is `this_node`
// Assumes that the array `triangles` is large enough to save all the triangles.
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

void build(std::vector<MBVHBuildNode>& mbvh, 
           const std::vector<BVHNode>& bvh, 
           const std::vector<CostNode>& cost_bvh, 
           const std::vector<u32>& triangle_index,
           u32 mbvh_parent, // the index of `mbvh` build node we are in
           u32 bvh_node, // the index of the current node in the `bvh` and `cost_bvh`
           u32 option) // which option to pick from the cost bvh
{
    const u8 meta = cost_bvh[bvh_node].meta[option];

    switch (meta & 0x3) {
    case CostNode::Leaf: {
        u32 triangles[3];
        const u32 triangle_count = get_triangles_from_subtree(bvh, bvh_node, triangles);
        if (triangle_count > 3) {
            printf("More than 3 triangles in leaf: %d\n", triangle_count);
        }

        //printf("LEAF, %d triangles\n", triangle_count);

        MBVHBuildNode& parent = mbvh[mbvh_parent];
        if (parent.child_count == 8) { 
            printf("parent already has 8 children!\n"); 
        }
        MBVHBuildNode::ChildNode& child = parent.child[parent.child_count++];

        child.type = MBVHBuildNode::ChildNode::Leaf;
        child.bounds = bvh[bvh_node].bounds;
        child.triangle_count = triangle_count;

        for (u32 i = 0; i < triangle_count; i++) {
            child.triangles[i] = triangle_index[triangles[i]];
        }
    } break;
    case CostNode::Internal: {
        MBVHBuildNode node;
        node.bounds = bvh[bvh_node].bounds;
        node.child_count = 0;
        for (u32 i = 0; i < 8; i++) {
            node.child[i].type = MBVHBuildNode::ChildNode::Empty;
        }

        const u32 mbvh_index = mbvh.size();
        mbvh.push_back(node);

        MBVHBuildNode& parent = mbvh[mbvh_parent];
        if (parent.child_count == 8) { 
            printf("parent already has 8 children!\n"); 
        }
        MBVHBuildNode::ChildNode& child = parent.child[parent.child_count++];
        child.type = MBVHBuildNode::ChildNode::Internal;
        child.bounds = bvh[bvh_node].bounds;
        child.internal_node_index = mbvh_index;

        const u32 left_option  = (meta >> 2) & 0x7;
        const u32 right_option = (meta >> 5) & 0x7;
        const u32 left  = bvh[bvh_node].left_first;
        const u32 right = left + 1;

        //printf("INTERNAL, leftopt: %d, rightopt: %d, left: %d, right: %d\n",
        //       left_option, right_option, left, right);

        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_index, left,  left_option);
        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_index, right, right_option);
    } break;
    case CostNode::Distribute: {
        const u32 left_option  = (meta >> 2) & 0x7;
        const u32 right_option = (meta >> 5) & 0x7;
        const u32 left  = bvh[bvh_node].left_first;
        const u32 right = left + 1;
                                                       
        //printf("DISTRIBUTE, leftopt: %d, rightopt: %d, left: %d, right: %d\n",
        //       left_option, right_option, left, right);
                                                      
        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_parent, left,  left_option);
        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_parent, right, right_option);
    } break;
    default: { dab_unreachable(); } break;
    }
}

// Terrible name...
struct BuildStep {
    u32 triangle_base_index;
    u32 child_base_index;
};

BuildStep build_mbvh_from_buildnodes(const std::vector<MBVHBuildNode>& build_nodes,
                                     std::vector<MBVH8Node>& mbvh,
                                     std::vector<u32>& triangle_order,
                                     u32 this_build_index,
                                     u32 this_mbvh_index,
                                     u32 triangle_base_index,
                                     u32 child_base_index)
{
    const MBVHBuildNode& build_node = build_nodes[this_build_index]; 
    MBVH8Node& node = mbvh[this_mbvh_index];

    // Count how much space we need to reserve for triangles and child nodes
    u32 internal_node_count = 0;
    u32 triangle_count = 0;
    u8 imask = 0;
    for (u32 i = 0; i < 8; i++) {
        switch (build_node.child[i].type) {
        case MBVHBuildNode::ChildNode::Leaf: {
            triangle_count += build_node.child[i].triangle_count;
        } break;
        case MBVHBuildNode::ChildNode::Internal: {
            internal_node_count++; 
            imask |= 1 << i;
        } break;
        case MBVHBuildNode::ChildNode::Empty: break;
        default: dab_unreachable(); break;
        }
    }

    // Reserve other MBVHNodes and triangles
    node.child_base_index    = child_base_index;
    node.triangle_base_index = triangle_base_index;

    node.origin = build_node.bounds.bmin;

    node.ex = cast(u8, 127.0f + ceil(log2((build_node.bounds.bmax.x - build_node.bounds.bmin.x) / 65535.0f)));
    node.ey = cast(u8, 127.0f + ceil(log2((build_node.bounds.bmax.y - build_node.bounds.bmin.y) / 65535.0f)));
    node.ez = cast(u8, 127.0f + ceil(log2((build_node.bounds.bmax.z - build_node.bounds.bmin.z) / 65535.0f)));

    node.imask = imask;

    const f32 fex = Float32(0, node.ex, 0).as_f32();
    const f32 fey = Float32(0, node.ey, 0).as_f32();
    const f32 fez = Float32(0, node.ez, 0).as_f32();

    u32 child_offset = 0;
    u32 triangle_offset = 0;
    for (u32 i = 0; i < 8; i++) {
        const Vector3 bmin = build_node.child[i].bounds.bmin - node.origin;
        const Vector3 bmax = build_node.child[i].bounds.bmax - node.origin;

        node.child[i].bminx = cast(u8, floor(bmin.x / fex));
        node.child[i].bminy = cast(u8, floor(bmin.y / fey));
        node.child[i].bminz = cast(u8, floor(bmin.z / fez));

        node.child[i].bmaxx = cast(u8, ceil(bmax.x / fex));
        node.child[i].bmaxy = cast(u8, ceil(bmax.y / fey));
        node.child[i].bmaxz = cast(u8, ceil(bmax.z / fez));

        switch (build_node.child[i].type) {
        case MBVHBuildNode::ChildNode::Leaf: {
            u8 meta = triangle_offset;
            for (u32 j = 0; j < build_node.child[i].triangle_count; j++) {
                triangle_order[triangle_base_index + triangle_offset + j] = build_node.child[i].triangles[j];
                meta |= 0b00100000 << j;
            }
            node.child[i].meta = meta;
            triangle_offset += build_node.child[i].triangle_count;
        } break;
        case MBVHBuildNode::ChildNode::Internal: {
            node.child[i].meta = 0b00100000 | (child_offset + 24);
            child_offset++;
        } break;
        case MBVHBuildNode::ChildNode::Empty: {
            node.child[i].meta = 0;
        } break;
        }
    }

    // !!! TODO !!!
    // Reorder all the child nodes in a better order

    // Recursively call for the children
    u32 child_index   = 0;
    u32 triangle_base = triangle_base_index + triangle_offset;
    u32 child_base    = child_base_index + child_offset;
    for (u32 i = 0; i < 8; i++) {
        if (build_node.child[i].type == MBVHBuildNode::ChildNode::Internal) {
            auto ret = build_mbvh_from_buildnodes(build_nodes,
                                                  mbvh,
                                                  triangle_order,
                                                  build_node.child[i].internal_node_index,
                                                  child_base_index + child_index,
                                                  triangle_base,
                                                  child_base);

            triangle_base = ret.triangle_base_index;
            child_base    = ret.child_base_index;
            child_index++;
        }
    }

    return { triangle_base, child_base };
}

MBVHBuildResult build_mbvh8(AABB const * aabbs, u32 aabb_count) {
    BVHBuildResult result = build_bvh(aabbs, aabb_count, 12, 1);

    const std::vector<BVHNode>& bvh = result.bvh;

    std::vector<CostNode> cost_bvh(bvh.size());
    calculate_cost(bvh, cost_bvh, 0);

    std::vector<MBVHBuildNode> build_mbvh;

    MBVHBuildNode root;
    root.bounds = bvh[0].bounds;
    root.child_count = 0;
    for (u32 i = 0; i < 8; i++) {
        root.child[i].type = MBVHBuildNode::ChildNode::Empty;
    }
    build_mbvh.push_back(root);
    
    if ((cost_bvh[0].meta[0] & 0x3) == CostNode::Internal) {
        const u32 left_option  = (cost_bvh[0].meta[0] >> 2) & 0x7;
        const u32 right_option = (cost_bvh[0].meta[0] >> 5) & 0x7;

        const u32 left  = bvh[0].left_first;
        const u32 right = left + 1;

        build(build_mbvh, bvh, cost_bvh, result.prim_order, 0, left,  left_option);
        build(build_mbvh, bvh, cost_bvh, result.prim_order, 0, right, right_option);
    } else {
        build(build_mbvh, bvh, cost_bvh, result.prim_order, 0, 0, 0);
    }
 
#if 1
    u32 mbvhprims = 0;
    u32 emptynodes = 0;
    for (u32 i = 0; i < build_mbvh.size(); i++) {
        if (build_mbvh[i].child_count == 0) { emptynodes++; }
        for (u32 j = 0; j < build_mbvh[i].child_count; j++) {
            mbvhprims += (build_mbvh[i].child[j].type == MBVHBuildNode::ChildNode::Leaf) ? 
                          build_mbvh[i].child[j].triangle_count : 0;
        }
    }

    printf("%llu build nodes constructed, with %d prims and %d empty nodes\n", build_mbvh.size(), mbvhprims, emptynodes);
#endif
    
    std::vector<MBVH8Node> mbvh(build_mbvh.size());
    std::vector<u32> prim_order(aabb_count);
    
    build_mbvh_from_buildnodes(build_mbvh,
                               mbvh,
                               prim_order,
                               0,
                               0,
                               0,
                               1);

    MBVHBuildResult ret;
    ret.mbvh = mbvh;
    ret.prim_order = prim_order;
    return ret;
}

std::vector<MBVH8Node> build_mbvh8_for_triangles_and_reorder(RenderTriangle* triangles, u32 triangle_count) {
    std::vector<AABB> aabbs(triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        aabbs[i] = AABB::for_triangle(triangles[i].v0,
                                      triangles[i].v1,
                                      triangles[i].v2);
    }

    MBVHBuildResult result = build_mbvh8(aabbs.data(), aabbs.size());

    printf("Reordering triangles...\n");
 
    // Reorder the triangles
    std::vector<RenderTriangle> tmp(triangles, triangles + triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        triangles[i] = tmp[result.prim_order[i]];
    }

    return result.mbvh;
}

