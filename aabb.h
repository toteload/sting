#ifndef GUARD_AABB_H
#define GUARD_AABB_H

struct alignas(16) AABB {
    alignas(16) vec3 bmin;
    alignas(16) vec3 bmax;

    // The union of two AABBs
    AABB merge(AABB other) const {
        return { { std::min(bmin.x, other.bmin.x), std::min(bmin.y, other.bmin.y), std::min(bmin.z, other.bmin.z) },
                 { std::max(bmax.x, other.bmax.x), std::max(bmax.y, other.bmax.y), std::max(bmax.z, other.bmax.z) } };
    }

    // Extend the AABB such that the given point is also contained
    AABB extend(vec3 p) const {
        return { { std::min(bmin.x, p.x), std::min(bmin.y, p.y), std::min(bmin.z, p.z) },
                 { std::max(bmax.x, p.x), std::max(bmax.y, p.y), std::max(bmax.z, p.z) } };
    }

    // The point in the middle of the AABB
    vec3 centroid() const {
        return 0.5f * bmin + 0.5f * bmax;
    }

    vec3 diagonal() const {
        return bmax - bmin;
    }

    float surface_area() const {
        const vec3 d = diagonal();
        return abs(2.0f * (d.x * d.y + d.x * d.z + d.y * d.z));
    }

    // Return the axis for which the AABB is largest
    uint32_t max_extend() const {
        const vec3 d = diagonal();
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

    __device__ bool intersect(vec3 ray_pos, vec3 ray_inv_dir, float* t_out) const {
        const vec3 t1 = (bmin - ray_pos) * ray_inv_dir;
        const vec3 t2 = (bmax - ray_pos) * ray_inv_dir;

        const vec3 emin = min_elements(t1, t2);
        const vec3 emax = max_elements(t1, t2);

        const float tmin = max(emin.x, max(emin.y, emin.z));
        const float tmax = min(emax.x, min(emax.y, emax.z));

        *t_out = tmin;

        return tmax > 0.0f && tmax > tmin;
    }

    static AABB empty() {
        return { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
    }

    static AABB for_triangle(vec3 p0, vec3 p1, vec3 p2) {
        return { { std::min({ p0.x, p1.x, p2.x }), std::min({ p0.y, p1.y, p2.y }), std::min({ p0.z, p1.z, p2.z }) },
                 { std::max({ p0.x, p1.x, p2.x }), std::max({ p0.y, p1.y, p2.y }), std::max({ p0.z, p1.z, p2.z }) } };
    }
};

#endif // GUARD_AABB_H
