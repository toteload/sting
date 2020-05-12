std::vector<Vector4> generate_sphere_mesh(u32 rows, u32 columns, f32 radius, Vector3 pos);

std::vector<Vector4> load_mesh_from_obj_file(const char* filename) {
    fastObjMesh* mesh = fast_obj_read(filename);
    if (!mesh) { 
        return {}; 
    }

    std::vector<Vector4> pos;

    for (u32 i = 0; i < mesh->group_count; i++) {
        fastObjGroup group = mesh->groups[i];

        u32 vi = 0;

        for (u32 j = 0; j < group.face_count; j++) {
            const u32 vertex_count = mesh->face_vertices[group.face_offset + j];

            if (vertex_count != 3) {
                // not a triangle, just skip it...
                continue;
            }

            for (u32 k = 0; k < vertex_count; k++) {
                fastObjIndex idx = mesh->indices[group.index_offset + vi];
                pos.push_back(vec4(mesh->positions[3 * idx.p + 0],
                                   mesh->positions[3 * idx.p + 1],
                                   mesh->positions[3 * idx.p + 2]));
                vi++;
            }
        }
    }

    return { pos, };
}

std::vector<Vector4> generate_sphere_mesh(u32 rows, u32 columns, f32 radius, Vector3 pos) {
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

    std::vector<Vector4> triangles;

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;

        triangles.push_back(vec4(pts[inext]));
        triangles.push_back(vec4(pts[i]));
        triangles.push_back(vec4(pts[top]));
    }

    for (u32 r = 0; r < rows - 2; r++) {
        for (u32 c = 0; c < columns; c++) {
            const u32 cnext = (c + 1) % columns;
            const u32 rowi = r * columns;
            const u32 rowinext = (r + 1) * columns;

            triangles.push_back(vec4(pts[rowi     + cnext]));
            triangles.push_back(vec4(pts[rowinext + c    ]));
            triangles.push_back(vec4(pts[rowi     + c    ]));

            triangles.push_back(vec4(pts[rowi     + cnext]));
            triangles.push_back(vec4(pts[rowinext + cnext]));
            triangles.push_back(vec4(pts[rowinext + c    ]));
        }
    }

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;
        const u32 rowi = (rows - 2) * columns;

        triangles.push_back(vec4(pts[rowi + i]));
        triangles.push_back(vec4(pts[rowi + inext]));
        triangles.push_back(vec4(pts[bottom]));
    }

    return triangles;
}
