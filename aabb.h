#ifndef GUARD_AABB_H
#define GUARD_AABB_H

struct alignas(16) AABB {
    alignas(16) Vector3 bmin;
    alignas(16) Vector3 bmax;

    // The union of two AABBs
    AABB merge(AABB other) const {
        return { vec3(min(bmin.x, other.bmin.x), min(bmin.y, other.bmin.y), min(bmin.z, other.bmin.z)),
                 vec3(max(bmax.x, other.bmax.x), max(bmax.y, other.bmax.y), max(bmax.z, other.bmax.z)), };
    }

    // Extend the AABB such that the given point is also contained
    AABB extend(Vector3 p) const {
        return { vec3(min(bmin.x, p.x), min(bmin.y, p.y), min(bmin.z, p.z)), 
                 vec3(max(bmax.x, p.x), max(bmax.y, p.y), max(bmax.z, p.z)), };
    }

    // The point in the middle of the AABB
    Vector3 centroid() const {
        return 0.5f * bmin + 0.5f * bmax;
    }

    Vector3 diagonal() const {
        return bmax - bmin;
    }

    f32 surface_area() const {
        const Vector3 d = diagonal();
        return abs(2.0f * (d.x * d.y + d.x * d.z + d.y * d.z));
    }

    // Return the axis for which the AABB is largest
    u32 max_extend() const {
        const Vector3 d = diagonal();
        if (d.x > d.y) {
            if (d.x > d.z) {
                return 0;
            } else {
                return 2;
            }
        } else {
            if (d.y > d.z) {
                return 1;
            } else {
                return 2;
            }
        }
    }

    __device__ f32 intersect(Vector3 ray_pos, Vector3 ray_inv_dir, f32* t_out) const {
        const Vector3 t1 = (bmin - ray_pos) * ray_inv_dir;
        const Vector3 t2 = (bmax - ray_pos) * ray_inv_dir;

        const Vector3 emin = Vector3::min(t1, t2);
        const Vector3 emax = Vector3::max(t1, t2);

        const f32 tmin = max(emin.x, max(emin.y, emin.z));
        const f32 tmax = min(emax.x, min(emax.y, emax.z));

        *t_out = tmin;

        return tmax > 0.0f && tmax >= tmin;
    }

    static AABB empty() {
        return { vec3(FLT_MAX, FLT_MAX, FLT_MAX), vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX), };
    }

    __host__ static AABB for_triangle(Vector3 p0, Vector3 p1, Vector3 p2) {
        return { vec3(min({ p0.x, p1.x, p2.x }), min({ p0.y, p1.y, p2.y }), min({ p0.z, p1.z, p2.z })),
                 vec3(max({ p0.x, p1.x, p2.x }), max({ p0.y, p1.y, p2.y }), max({ p0.z, p1.z, p2.z })), };
    }
};

#endif // GUARD_AABB_H
