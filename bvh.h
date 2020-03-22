#ifndef GUARD_BVH_H
#define GUARD_BVH_H

#include <vector>

#include "aabb.h"

struct alignas(16) BVHNode {
    union {
        AABB bounds;

        // - bmin and bmax are the min and max of the bounding box of this node.
        // - count is the amount of primitives in this leaf node, if this is 0 then
        // this node is an interior node
        // - left_first is the index of the left child of this node if this is an
        // interior node, the index of the right child is left_first + 1 or if this
        // is a leaf node then left_first is the index of the first primitive
        // in the primitive array for this node.
        struct {
            vec3 bmin; uint32_t count;
            vec3 bmax; uint32_t left_first;
        };
    };

    BVHNode() { }

    __device__ bool is_leaf() const { return count > 0; }
};

// Just some sanity checking
static_assert(offsetof(BVHNode, bmin) == 0, "");
static_assert(offsetof(BVHNode, count) == 12, "");
static_assert(offsetof(BVHNode, bmax) == 16, "");
static_assert(offsetof(BVHNode, left_first) == 28, "");
static_assert(sizeof(BVHNode) == 32, "");

struct alignas(16) MBVHNode {
    AABB4 aabbs;
    u32 count[4];
    u32 left_first[4];

    void set_aabb(const AABB& bounds, u32 index) {
        aabbs.bminx[index] = bounds.bmin.x;
        aabbs.bminy[index] = bounds.bmin.y;
        aabbs.bminz[index] = bounds.bmin.z;

        aabbs.bmaxx[index] = bounds.bmax.x;
        aabbs.bmaxy[index] = bounds.bmax.y;
        aabbs.bmaxz[index] = bounds.bmax.z;
    }

    bool is_leaf(u32 index) const {
        return count[index] == 0;
    }
};

enum RenderMaterial : u32 {
    MATERIAL_DIFFUSE = 1,
    MATERIAL_MIRROR = 2,
    MATERIAL_EMISSIVE = 3,
};

// NOTE
// This triangle format has not yet been optimized for size.
// It is the way it is for the sake of simplicity
struct alignas(16) RenderTriangle {
    vec3 v0; f32 colorr;
    vec3 v1; f32 colorg;
    vec3 v2; f32 colorb;
    vec3 n0; u32 material;
    vec3 n1; f32 light_intensity;
    vec3 n2; f32 area;
    vec3 face_normal; u32 pad6;

    RenderTriangle(vec3 v0, vec3 v1, vec3 v2) :
        v0(v0), v1(v1), v2(v2), 
        material(MATERIAL_DIFFUSE), 
        colorr(1.0f), colorg(1.0f), colorb(1.0f), 
        light_intensity(1.0f),
        area(triangle_area(v0, v1, v2))
    {
        n0 = n1 = n2 = face_normal = triangle_normal(v0, v1, v2);
    }

    RenderTriangle(vec3 v0, vec3 v1, vec3 v2, 
                   vec3 n0, vec3 n1, vec3 n2) : 
        v0(v0), v1(v1), v2(v2), 
        n0(n0), n1(n1), n2(n2), 
        material(MATERIAL_DIFFUSE), 
        colorr(1.0f), colorg(1.0f), colorb(1.0f), 
        light_intensity(1.0f),
        area(triangle_area(v0, v1, v2))
    { 
        face_normal = triangle_normal(v0, v1, v2);
    }

    __host__ __device__ vec3 color() const {
        return vec3(colorr, colorg, colorb);
    }
};

struct Transform {
    vec3 scale;
    vec3 offset;
    // currently no rotation
};

struct MeshIndex {
    u32 bvh_offset;
    u32 triangle_offset;

    u32 bvhnode_count;
    u32 triangle_count;

    const char* name;
};

struct RenderInstance {
    Transform transform;
    MeshIndex mesh;
};

#ifndef __CUDACC__

// NOTE
// This will reorder the `triangles` array
std::vector<BVHNode> build_bvh_for_triangles(RenderTriangle* triangles, uint32_t triangle_count);

void build_mbvh(MBVHNode& mnode, std::vector<MBVHNode>& nodes, BVHNode const * bvh, u32 current_node) {
    const BVHNode& current = bvh[current_node];

    const BVHNode& left  = bvh[current.left_first];
    const BVHNode& right = bvh[current.left_first+1];

    mnode.set_aabb(left.bounds, 0);
    mnode.set_aabb(right.bounds, 1);
    mnode.set_aabb(AABB::empty(), 2);
    mnode.set_aabb(AABB::empty(), 3);

    mnode.count[0] = left.count;
    mnode.count[1] = right.count;

    const u32 child_index = nodes.size();
    const bool has_left_child = !left.is_leaf();

    u32 node_offset = child_index;

    if (!left.is_leaf()) {
        nodes.push_back({});
        mnode.left_first[0] = node_offset++;
    } else {
        mnode.left_first[0] = left.left_first;
    }

    if (!right.is_leaf()) {
        nodes.push_back({});
        mnode.left_first[1] = node_offset++;
    } else {
        mnode.left_first[1] = right.left_first;
    }

    if (!left.is_leaf()) {
        build_mbvh(nodes[child_index], nodes, bvh, current.left_first);
    }

    if (!right.is_leaf()) {
        build_mbvh(nodes[child_index + ((has_left_child) ? 1 : 0)], nodes, bvh, current.left_first+1);
    }
}

template<typename T, u32 Capacity>
struct Stack {
    T data[Capacity];
    u32 top;

    Stack() : top(0) { }
    __device__ Stack(std::initializer_list<T> l) : top(l.size()) {
        for (u32 i = 0; i < l.size(); i++) {
            data[i] = l.begin()[i];
        }
    }

    __device__ bool is_empty() const { return top == 0; }
    u32 capacity() const { return Capacity; }
    u32 size() const { return top; }
    __device__ void push(T v) { data[top++] = v; }
    __device__ T pop() { return data[--top]; }
    T peek() const { return data[top-1]; }
};

void collapse_mbvh(std::vector<MBVHNode>& mbvh) {
    Stack<u32, 128> stack({ 0 });

    while (!stack.is_empty()) {
        MBVHNode& node = mbvh[stack.pop()];
        if (!node.is_leaf(0)) {
            stack.push(node.left_first[0]);
        } else {
            
        }

        if (!node.is_leaf(1)) {
            stack.push(node.left_first[1]);
        }


    }
}

std::vector<MBVHNode> build_mbvh_from_bvh(BVHNode const * bvh, u32 bvh_node_count) {
    std::vector<MBVHNode> mbvh;
    mbvh.reserve(bvh_node_count);

    mbvh.push_back({});
    MBVHNode& root = mbvh.back();
    
    build_mbvh(root, mbvh, bvh, 0);

    return mbvh;
}

#if 1

struct Scene {
    std::vector<BVHNode> instance_bvh;
    std::vector<RenderInstance> instances;

    std::vector<BVHNode> mesh_bvh;
    std::vector<RenderTriangle> mesh_triangles;

    // We store the MeshIndex to keep track of all the stored meshes
    std::vector<MeshIndex> mesh;
    std::vector<char> name_buffer;

    // materials ?

    MeshIndex register_mesh(RenderTriangle const * triangles, u32 triangle_count, const char* name) {
        const u32 triangle_offset = mesh_triangles.size();
        mesh_triangles.insert(mesh_triangles.end(), triangles, triangles + triangle_count);

        const std::vector<BVHNode> bvh = build_bvh_for_triangles(mesh_triangles.data() + triangle_offset, triangle_count);

        const u32 bvh_offset = mesh_bvh.size();
        mesh_bvh.insert(mesh_bvh.end(), bvh.begin(), bvh.end());

        const u32 name_offset = name_buffer.size();
        name_buffer.resize(name_buffer.size() + 64);
        snprintf(&name_buffer[name_offset], 64, "%s", name);

        const MeshIndex id = { .bvh_offset = bvh_offset, 
                               .triangle_offset = triangle_offset, 
                               .name = &name_buffer[name_offset] };
        return id;
    }

    void register_instance(MeshIndex id, const Transform& transform) {
        instances.push_back({ .transform = transform, .mesh = id });
    }

    void build_top_level_bvh() {
    }
};
#endif

#endif

#ifdef __CUDACC__

struct alignas(16) BVHTriangleIntersection {
    u32 id; // the most significant bit will be set if there is no hit
    f32 t, u, v;

    __device__ bool hit() const { return id == UINT32_MAX; }
};

struct BVHInstanceIntersection {
    u32 instance_id; u32 triangle_id;
    f32 t, u, v;

    __device__ bool hit() const { return instance_id == UINT32_MAX; }
};

__device__ BVHInstanceIntersection bvh_intersect_instance_triangles(BVHNode const * instance_bvh, 
                                                                    RenderInstance const * instances,
                                                                    BVHNode const * mesh_bvh, 
                                                                    RenderTriangle const * triangles,
                                                                    Ray ray);

__device__ BVHTriangleIntersection bvh_intersect_triangles(BVHNode const * bvh, 
                                                           RenderTriangle const * triangles, 
                                                           Ray ray);

__device__ bool bvh_intersect_triangles_shadowcast(BVHNode const * bvh, RenderTriangle const * triangles, 
                                                   Ray ray);
#endif

#endif // GUARD_BVH_H
