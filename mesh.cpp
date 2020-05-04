std::vector<RenderTriangle> generate_sphere_mesh(u32 rows, u32 columns, f32 radius, Vector3 pos, u32 material_id, bool add_normals=false);
std::vector<RenderTriangle> load_mesh_from_obj_file(const char* filename, u32 material_id);

std::vector<RenderTriangle> generate_sphere_mesh(u32 rows, u32 columns, f32 radius, Vector3 pos, u32 material_id, bool add_normals) {
    if (rows < 2 || columns < 3) {
        return { };
    }

    std::vector<Vector3> pts;
    std::vector<Vector3> normals;

    for (u32 i = 1; i < rows; i++) {
        for (u32 j = 0; j < columns; j++) {
            const f32 phi = i * M_PI / cast(f32, rows);
            const f32 theta = (cast(f32, j) / cast(f32, columns)) * 2.0f * M_PI;
            const Vector3 n = spherical_to_cartesian(phi, theta);
            pts.push_back(pos + radius * n);
            normals.push_back(n);
        }
    }

    const size_t top = pts.size();
    pts.push_back(pos + radius * vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));

    const size_t bottom = pts.size();
    pts.push_back(pos + radius * vec3(0.0f, -1.0f, 0.0f));
    normals.push_back(vec3(0.0f, -1.0f, 0.0f));

    std::vector<RenderTriangle> triangles;

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;

        if (add_normals) {
            triangles.push_back(RenderTriangle(pts[inext], pts[i], pts[top], 
                                               normals[inext], normals[i], normals[top], 
                                               material_id));
        } else {
            triangles.push_back(RenderTriangle(pts[inext], pts[i], pts[top], material_id));
        }
    }

    for (u32 r = 0; r < rows - 2; r++) {
        for (u32 c = 0; c < columns; c++) {
            const u32 cnext = (c + 1) % columns;
            const u32 rowi = r * columns;
            const u32 rowinext = (r + 1) * columns;

            if (add_normals) {
                triangles.push_back(RenderTriangle(pts[rowi + cnext], pts[rowinext + c], pts[rowi + c], 
                                                   normals[rowi + cnext], normals[rowinext +c], normals[rowi + c], 
                                                   material_id));
                triangles.push_back(RenderTriangle(pts[rowi + cnext], pts[rowinext + cnext], pts[rowinext + c], 
                                                   normals[rowi + cnext], normals[rowinext + cnext], normals[rowinext + c], 
                                                   material_id));
            } else {
                triangles.push_back(RenderTriangle(pts[rowi + cnext], pts[rowinext + c], pts[rowi + c], 
                                                   material_id));
                triangles.push_back(RenderTriangle(pts[rowi + cnext], pts[rowinext + cnext], pts[rowinext + c], 
                                                   material_id));
            }
        }
    }

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;
        const u32 rowi = (rows - 2) * columns;

        if (add_normals) {
            triangles.push_back(RenderTriangle(pts[rowi + i], pts[rowi + inext], pts[bottom], 
                                               normals[rowi + i], normals[rowi + inext], normals[bottom],
                                               material_id));
        } else {
            triangles.push_back(RenderTriangle(pts[rowi + i], pts[rowi + inext], pts[bottom], material_id));
        }
    }

    return triangles;
}

std::vector<RenderTriangle> load_mesh_from_obj_file(const char* filename, u32 material_id) {
    fastObjMesh* mesh = fast_obj_read(filename);

    if (!mesh) {
        return { };
    }

    std::vector<RenderTriangle> triangles;

    // Loop over all the groups in the mesh
    for (uint32_t i = 0; i < mesh->group_count; i++) {
        fastObjGroup group = mesh->groups[i];

        uint32_t vertex_index = 0;

        // Loop over all the faces in this group
        for (uint32_t j = 0; j < group.face_count; j++) {
            const uint32_t vertex_count = mesh->face_vertices[group.face_offset + j];

            if (vertex_count != 3) {
                // This is a face that is not a triangle. Oh oh...
                // TODO report this or triangulize this face or something
                // for now just skip it
                continue;
            }

            Vector3 vertices[3];

            // Loop over all the vertices in this face
            for (uint32_t k = 0; k < vertex_count; k++) {
                fastObjIndex index = mesh->indices[group.index_offset + vertex_index];

                vertices[k] = vec3(mesh->positions[3 * index.p + 0], 
                                   mesh->positions[3 * index.p + 1], 
                                   mesh->positions[3 * index.p + 2]);

                vertex_index++;
            }

            triangles.push_back(RenderTriangle(vertices[0], vertices[1], vertices[2], material_id));
        }
    }

    fast_obj_destroy(mesh);

    return triangles;
}
