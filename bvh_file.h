#pragma once

struct BVHFileHeader {
    u32 magic;
    u32 triangle_count;
    u32 node_count;
    u32 bvh_depth;
};

void save_bvh_object(RenderTriangle const * triangles, u32 triangle_count,
                     u32 partition_bin_count, u32 max_prims_in_leaf);

struct BVHObject {
    u32 valid;
    std::vector<BVHNode>        bvh;
    std::vector<RenderTriangle> triangles;
};

BVHObject load_bvh_object(void const * data);
