struct Mesh {
    std::vector<RenderTriangle> triangles;
};

Mesh load_mesh(const char* filename, bool lerp_normals) {
    fastObjMesh* mesh = fast_obj_read(filename);
    if (!mesh) {
        // oh oh
    }

    std::vector<vec3> normals;
    normals.reserve(mesh->position_count);

    bool mesh_has_normals = false;

    for (u32 i = 0; i < mesh->group_count; i++) {
        fastObjGroup group = mesh->groups[i];
        u32 vertex_index = 0;

        std::vector<vec3> positions;

        for (u32 j = 0; j < group.face_count; j++) { 
            const u32 vertex_count = mesh->face_vertices[group.face_offset + j];

            if (vertex_count != 3) {
                // oh oh
            }

            vec3 vertices[3];
            vec3 normals[3];


        }
    }
}
