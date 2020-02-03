#ifndef GUARD_AABB_H
#define GUARD_AABB_H

struct alignas(16) AABB {
    alignas(16) vec3 min;
    alignas(16) vec3 max;

    // The union of two AABBs
    AABB merge(AABB other) const {
        return { { std::min(min.x, other.min.x), std::min(min.y, other.min.y), std::min(min.z, other.min.z) },
                 { std::max(max.x, other.max.x), std::max(max.y, other.max.y), std::max(max.z, other.max.z) } };
    }

    // Extend the AABB such that the given point is also contained
    AABB extend(vec3 p) const {
        return { { std::min(min.x, p.x), std::min(min.y, p.y), std::min(min.z, p.z) },
                 { std::max(max.x, p.x), std::max(max.y, p.y), std::max(max.z, p.z) } };
    }

    vec3 centroid() const {
        return 0.5f * min + 0.5f * max;
    }

    vec3 diagonal() const {
        return max - min;
    }

    float surface_area() const {
        const vec3 d = diagonal();
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
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

    static AABB empty() {
        return { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
    }

    static AABB for_triangle(vec3 p0, vec3 p1, vec3 p2) {
        return { { std::min({ p0.x, p1.x, p2.x }), std::min({ p0.y, p1.y, p2.y }), std::min({ p0.z, p1.z, p2.z }) },
                 { std::max({ p0.x, p1.x, p2.x }), std::max({ p0.y, p1.y, p2.y }), std::max({ p0.z, p1.z, p2.z }) } };
    }
};

#endif // GUARD_AABB_H
