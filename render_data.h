#pragma once

enum RenderMaterial : u32 {
    MATERIAL_DIFFUSE = 1,
    MATERIAL_MIRROR = 2,
    MATERIAL_EMISSIVE = 3,
};

struct Material {
    enum Type : u32 {
        DIFFUSE,
        MIRROR,
        EMISSIVE,
    } type;

    f32 r, g, b; f32 light_intensity;

    __device__ Vector3 color() const { return vec3(r, g, b); }
};

struct alignas(16) RenderTriangle {
    Vector3 v0; u32 n0;
    Vector3 v1; u32 n1;
    Vector3 v2; u32 n2;
    Vector3 face_normal; u32 material_id;

    RenderTriangle(Vector3 v0, Vector3 v1, Vector3 v2, u32 material_id);
    RenderTriangle(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 n0, Vector3 n1, Vector3 n2, u32 material_id);
};

struct alignas(16) RenderSphere {
    Vector3 pos; f32 radius;
    u32 material_id; u32 pad[3];
};

inline RenderTriangle::RenderTriangle(Vector3 v0, Vector3 v1, Vector3 v2, u32 material_id) :
    v0(v0), n0(pack_normal(vec3(0.0f, 1.0f, 0.0f))),
    v1(v1), n1(pack_normal(vec3(0.0f, 1.0f, 0.0f))),
    v2(v2), n2(pack_normal(vec3(0.0f, 1.0f, 0.0f))),
    face_normal(triangle_normal(v0, v1, v2)), material_id(material_id)
{ }

inline RenderTriangle::RenderTriangle(Vector3 v0, Vector3 v1, Vector3 v2, 
                                      Vector3 n0, Vector3 n1, Vector3 n2,
                                      u32 material_id) : 
    v0(v0), n0(pack_normal(n0)),
    v1(v1), n1(pack_normal(n1)),
    v2(v2), n2(pack_normal(n2)),
    face_normal(triangle_normal(v0, v1, v2)), material_id(material_id)
{ }                                                  
                               
