#pragma once

struct BVHFileHeader {
    u32 magic;
    u32 triangle_count;
    u32 node_count;
};

constexpr u32 BVHFILEHEADER_MAGIC_VALUE = 0x7073104D;
 
struct BVHObject {
    u32 valid;
    std::vector<BVHNode>        bvh;
    std::vector<RenderTriangle> triangles;
};

bool save_bvh_object(const char* filename,
                     BVHNode const * nodes, u32 node_count,
                     RenderTriangle const * triangles, u32 triangle_count)
{
    FILE* f = fopen(filename, "wb");
    if (!f) { return false; }

    BVHFileHeader header;
    header.magic = BVHFILEHEADER_MAGIC_VALUE;
    header.triangle_count = triangle_count;
    header.node_count = node_count;

    fwrite(&header, 1, sizeof(header), f);
    fwrite(nodes, 1, node_count * sizeof(BVHNode), f);
    fwrite(triangles, 1, triangle_count * sizeof(RenderTriangle), f);

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
    RenderTriangle const * triangles = cast(RenderTriangle*, nodes + header->node_count);

    obj.bvh.resize(header->node_count);
    obj.triangles.resize(header->triangle_count);
    memcpy(obj.bvh.data(), nodes, header->node_count * sizeof(BVHNode));
    memcpy(obj.triangles.data(), triangles, header->triangle_count * sizeof(RenderTriangle));
    obj.valid = true;

    return obj;
}
