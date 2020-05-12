#include "bvh.h"
#include "dab/dab.h"
#include <math.h>

union alignas(16) BVHBuildPrimitive {
    AABB bounds;

    struct {
        alignas(16) Vector3 bmin; u32 index;
        alignas(16) Vector3 bmax;
    };

    BVHBuildPrimitive() { }
    BVHBuildPrimitive(const AABB& bounds, u32 index) : bmin(bounds.bmin), index(index), bmax(bounds.bmax) { }
    BVHBuildPrimitive(const Vector3& bmin, const Vector3& bmax, u32 index) : bmin(bmin), index(index), bmax(bmax) { }
};

static_assert(sizeof(BVHBuildPrimitive) == sizeof(AABB), "sizes should be the same");

struct PartitionResult {
    bool should_split;
    u32 left_count;
};

PartitionResult partition(BVHBuildPrimitive* aabbs, AABB node_bounds, u32 start, u32 end, u32 bin_count) {
    f32 lowest_cost = FLT_MAX;
    f32 split_coord; 
	u32 split_axis;

    for (u32 bin = 1; bin < bin_count; bin++) {
        const f32 offset = cast(f32, bin) / bin_count;

        for (u32 axis = 0; axis < 3; axis++) {
            const f32 split = node_bounds.bmin[axis] + offset * (node_bounds.bmax[axis] - node_bounds.bmin[axis]);

            u32 left_count = 0, right_count = 0;
            AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();

            for (u32 i = start; i < end; i++) {
                if (aabbs[i].bounds.centroid()[axis] <= split) {
                    left_bounds = left_bounds.merge(aabbs[i].bounds);
                    left_count++;
                } else {
                    right_bounds = right_bounds.merge(aabbs[i].bounds);
                    right_count++;
                }
            }

            // Need to do this in the case that all primitives are on one side, which causes the AABB of
            // the other side to be empty and return a surface area = inf. inf * 0.0f = nan, so we cannot
            // just multiply.
			const f32 cost = ((left_count)  ? (left_count  * left_bounds.surface_area())  : 0.0f) +
							 ((right_count) ? (right_count * right_bounds.surface_area()) : 0.0f);            

			if (cost < lowest_cost) {
                lowest_cost = cost;
                split_coord = split;
                split_axis = axis;
            }
        }
    }

    const f32 current_node_cost = node_bounds.surface_area() * (end - start);
    if (current_node_cost <= lowest_cost) {
        return { false, 0 };
    }

    u32 first_right = start;

    for (u32 i = start; i < end; i++) {
        if (aabbs[i].bounds.centroid()[split_axis] <= split_coord) {
            std::swap(aabbs[first_right], aabbs[i]);
            first_right++;
        }
    }

    return { true , first_right - start };
}

u32 build_recursive(BVHBuildPrimitive* aabbs, std::vector<BVHNode>& nodes, 
                    u32 start, u32 end, 
                    u32 this_node, u32 depth,
                    u32 partition_bin_count, u32 max_prims_in_leaf) 
{
    const u32 count = end - start;

    if (count <= max_prims_in_leaf) {
        // This is a leaf node
        nodes[this_node].count = count;
        nodes[this_node].left_first = start;
        return depth;
    }

    const PartitionResult result = partition(aabbs, nodes[this_node].bounds, start, end, partition_bin_count);

    u32 mid;
    if (!result.should_split) {
        // The partitioning says we should not split since it would increase cost, but we still have more
        // primitives than we would like to have for this node to be a leaf, so we split anyway.
        // The splitting we do evenly.
        mid = start + count / 2;
    } else {
        mid = start + result.left_count;
    }

    const u32 left = nodes.size();

    nodes[this_node].count = 0;
    nodes[this_node].left_first = left;

    // Reserve a left and right node
    nodes.push_back({});
    nodes.push_back({});

    AABB left_bounds = AABB::empty(), right_bounds = AABB::empty();
    for (u32 i = start; i < mid; i++) {
        left_bounds = left_bounds.merge(aabbs[i].bounds);
    }

    for (u32 i = mid; i < end; i++) {
        right_bounds = right_bounds.merge(aabbs[i].bounds);
    }

    nodes[left  ].bounds = left_bounds;
    nodes[left+1].bounds = right_bounds;

    const u32 depth_left = build_recursive(aabbs, nodes, 
                                           start, mid, 
                                           left, depth + 1, 
                                           partition_bin_count, max_prims_in_leaf);
    const u32 depth_right = build_recursive(aabbs, nodes, 
                                            mid, end, 
                                            left + 1, depth + 1, 
                                            partition_bin_count, max_prims_in_leaf);

    return max(depth_left, depth_right);
}

BVHBuildResult build_bvh(AABB const * aabbs, u32 aabb_count, u32 partition_bin_count, u32 max_prims_in_leaf) {
    std::vector<BVHBuildPrimitive> build_prims(aabb_count);
    AABB root_bounds = AABB::empty();
    for (u32 i = 0; i < aabb_count; i++) {
        build_prims[i] = BVHBuildPrimitive(aabbs[i], i);
        root_bounds = root_bounds.merge(aabbs[i]);
    }

    std::vector<BVHNode> nodes;
    nodes.push_back({});
    nodes[0].bounds = root_bounds;

    build_recursive(build_prims.data(), nodes,
                    0, aabb_count,
                    0, 0,
                    partition_bin_count, max_prims_in_leaf);

    if (nodes.size() > 1) {
        nodes[0].count = 0;
        nodes[0].left_first = 1;
    }

    std::vector<u32> prim_order(aabb_count);
    for (u32 i = 0; i < aabb_count; i++) {
        prim_order[i] = build_prims[i].index;
    }

    return { .bvh        = nodes,
             .prim_order = prim_order, };
}

std::vector<BVHNode> build_bvh_for_triangles_and_reorder(std::vector<Vector4>& triangles, 
                                                         u32 partition_bin_count, 
                                                         u32 max_primitives_in_leaf) 
{
    const u32 triangle_count = triangles.size() / 3;
    std::vector<AABB> aabbs(triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        aabbs[i] = AABB::for_triangle(vec3(triangles[i * 3    ]),
                                      vec3(triangles[i * 3 + 1]),
                                      vec3(triangles[i * 3 + 2]));
    }

    BVHBuildResult result = build_bvh(aabbs.data(), triangle_count, partition_bin_count, max_primitives_in_leaf);

    // Reorder the triangles
    std::vector<Vector4> tmp(triangles);
    for (u32 i = 0; i < triangle_count; i++) {
        triangles[i * 3    ] = tmp[result.prim_order[i] * 3    ];
        triangles[i * 3 + 1] = tmp[result.prim_order[i] * 3 + 1];
        triangles[i * 3 + 2] = tmp[result.prim_order[i] * 3 + 2];
    }
    
    return result.bvh;
}

CBVH compress_bvh(std::vector<BVHNode> bvh) {
    const Vector3 origin = bvh[0].bounds.bmin;

    const u8 ex = cast(u8, 127.0f + ceil(log2((bvh[0].bounds.bmax.x - bvh[0].bounds.bmin.x) / 65535.0f)));
    const u8 ey = cast(u8, 127.0f + ceil(log2((bvh[0].bounds.bmax.y - bvh[0].bounds.bmin.y) / 65535.0f)));
    const u8 ez = cast(u8, 127.0f + ceil(log2((bvh[0].bounds.bmax.z - bvh[0].bounds.bmin.z) / 65535.0f)));

    const f32 fex = Float32(0, ex, 0).as_f32();
    const f32 fey = Float32(0, ey, 0).as_f32();
    const f32 fez = Float32(0, ez, 0).as_f32();

    std::vector<CBVHNode> cbvh;
    cbvh.reserve(bvh.size());
    
    for (u32 i = 0; i < bvh.size(); i++) {
        const Vector3 bmin = bvh[i].bmin - origin;
        const Vector3 bmax = bvh[i].bmax - origin;

        const u16 bminx = cast(u16, floor(bmin.x / fex));
        const u16 bminy = cast(u16, floor(bmin.y / fey));
        const u16 bminz = cast(u16, floor(bmin.z / fez));

        const u16 bmaxx = cast(u16, ceil(bmax.x / fex));
        const u16 bmaxy = cast(u16, ceil(bmax.y / fey));
        const u16 bmaxz = cast(u16, ceil(bmax.z / fez));

        const u32 meta = (bvh[i].count << 28) | (bvh[i].left_first & 0x0fffffff);

        cbvh.push_back({ bminx, bminy, bminz, bmaxx, bmaxy, bmaxz, meta });
    }

    return { .data = { origin, ex, ey, ez, 0, }, .nodes = cbvh };
}
 
struct BVHFileHeader {
    u32 magic;
    u32 triangle_vertex_count;
    u32 node_count;
};

constexpr u32 BVHFILEHEADER_MAGIC_VALUE = 0x7073104D;

bool save_bvh_object(const char* filename,
                     BVHNode const * nodes, u32 node_count,
                     Vector4 const * triangles, u32 triangle_vertex_count)
{
    FILE* f = fopen(filename, "wb");
    if (!f) { return false; }

    BVHFileHeader header;
    header.magic = BVHFILEHEADER_MAGIC_VALUE;
    header.triangle_vertex_count = triangle_vertex_count;
    header.node_count = node_count;

    fwrite(&header, 1, sizeof(header), f);
    fwrite(nodes, 1, node_count * sizeof(BVHNode), f);
    fwrite(triangles, 1, triangle_vertex_count * sizeof(Vector4), f);

    fclose(f);

    return true;
}

BVHObject load_bvh_object(void const * data) {
    BVHObject obj;
    obj.valid = false;

    BVHFileHeader* header = cast(BVHFileHeader*, data);
    if (header->magic != BVHFILEHEADER_MAGIC_VALUE) {
        return obj;
    }

    BVHNode const * nodes = cast(BVHNode*, header + 1);
    Vector4 const * triangles = cast(Vector4*, nodes + header->node_count);

    obj.bvh.resize(header->node_count);
    obj.triangles.resize(header->triangle_vertex_count);
    memcpy(obj.bvh.data(), nodes, header->node_count * sizeof(BVHNode));
    memcpy(obj.triangles.data(), triangles, header->triangle_vertex_count * sizeof(Vector4));
    obj.valid = true;

    return obj;
}
 
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

        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_index, left,  left_option);
        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_index, right, right_option);
    } break;
    case CostNode::Distribute: {
        const u32 left_option  = (meta >> 2) & 0x7;
        const u32 right_option = (meta >> 5) & 0x7;
        const u32 left  = bvh[bvh_node].left_first;
        const u32 right = left + 1;
                                                       
        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_parent, left,  left_option);
        build(mbvh, bvh, cost_bvh, triangle_index, mbvh_parent, right, right_option);
    } break;
    default: { DAB_UNREACHABLE(); } break;
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
        default: DAB_UNREACHABLE(); break;
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

std::vector<MBVH8Node> build_mbvh8_for_triangles_and_reorder(std::vector<Vector4> triangles) {
    const u32 triangle_count = triangles.size() / 3;
    std::vector<AABB> aabbs(triangle_count);
    for (u32 i = 0; i < triangle_count; i++) {
        aabbs[i] = AABB::for_triangle(vec3(triangles[i * 3    ]),
                                      vec3(triangles[i * 3 + 1]),
                                      vec3(triangles[i * 3 + 2]));
    }

    MBVHBuildResult result = build_mbvh8(aabbs.data(), aabbs.size());

    printf("Reordering triangles...\n");
 
    // Reorder the triangles
    std::vector<Vector4> tmp(triangles);
    for (u32 i = 0; i < triangle_count; i++) {
        triangles[i * 3    ] = tmp[result.prim_order[i * 3    ]];
        triangles[i * 3 + 1] = tmp[result.prim_order[i * 3 + 1]];
        triangles[i * 3 + 2] = tmp[result.prim_order[i * 3 + 2]];
    }

    return result.mbvh;
}

