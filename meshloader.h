#if 0
struct TriangleRef {
    u32 i[3];
};

std::vector<RenderTriangle> load_mesh(const char* filename) {
    fastObjMesh* mesh = fast_obj_read(filename);
    if (!mesh) {
        // oh oh
    }

    std::vector<vec3> positions;
    positions.resize(mesh->position_count);

    std::vector<vec3> normals;
    normals.resize(mesh->position_count, vec3(0.0f));

    std::vector<u32> count;
    count.resize(mesh->position_count, 0);

    std::vector<TriangleRef> tris;
    tris.reserve(mesh->face_count);

    // Loop over all the groups in the mesh
    for (u32 i = 0; i < mesh->group_count; i++) {
        fastObjGroup group = mesh->groups[i];
        u32 vertex_index = 0;


        // Loop over all the faces in the group
        for (u32 j = 0; j < group.face_count; j++) { 
            const u32 vertex_count = mesh->face_vertices[group.face_offset + j];

            if (vertex_count != 3) {
                // oh oh
            }

            vec3 vs[3];
            TriangleRef tri;

            // Loop over all the points in this face to find their positions
            for (u32 k = 0; k < vertex_count; k++) {
                const fastObjIndex index = mesh->indices[group.index_offset + vertex_index + k];

                const vec3 v = vec3(mesh->positions[3 * index.p + 0], 
                                    mesh->positions[3 * index.p + 1], 
                                    mesh->positions[3 * index.p + 2]);
                vs[k] = v;
                tri.i[k] = index.p;
                positions[index.p] = v;
            }

            tris.push_back(tri);

            const vec3 n = triangle_normal(vs[0], vs[1], vs[2]);

            for (u32 k = 0; k < vertex_count; k++) {
                const fastObjIndex index = mesh->indices[group.index_offset + vertex_index + k];

                normals[index.p] += n;
                count[index.p]++;
            }

            vertex_index += vertex_count;
        }
    }

    for (u32 i = 0; i < normals.size(); i++) {
        normals[i] = 1.0f / cast(f32, count[i]) * normals[i];
    }

    std::vector<RenderTriangle> triangles;
    triangles.reserve(mesh->face_count);

    for (u32 i = 0; i < tris.size(); i++) {
        triangles.push_back(RenderTriangle(positions[tri[i].i[0]], positions[tri[i].i[1]], positions[tri[i].i[2]],
                                           normals[tri[i].i[0]],   normals[tri[i].i[1]],   normals[tri[i].i[2]]));
    }

    return triangles;
}
#endif
