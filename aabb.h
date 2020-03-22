#ifndef GUARD_AABB_H
#define GUARD_AABB_H

#include "stingmath.h"

#if 1
struct alignas(16) AABB4 {
    struct Intersection {
        vec4 t;

        u32 hit_mask() const {
            return ((f32_to_bits(t[3]) & 0x1) << 3) | 
                   ((f32_to_bits(t[2]) & 0x1) << 2) | 
                   ((f32_to_bits(t[1]) & 0x1) << 1) | 
                   ((f32_to_bits(t[0]) & 0x1)     );
        }
    };

    alignas(16) vec4 bminx, bminy, bminz;
    alignas(16) vec4 bmaxx, bmaxy, bmaxz;

    Intersection intersect(vec3 ray_pos, vec3 ray_inv_dir) {
        const vec4 t1x = ray_inv_dir.x * (bminx - vec4(ray_pos.x));
        const vec4 t2x = ray_inv_dir.x * (bmaxx - vec4(ray_pos.x));
        const vec4 minx = vec4::min(t1x, t2x);
        const vec4 maxx = vec4::max(t1x, t2x);

        const vec4 t1y = ray_inv_dir.y * (bminy - vec4(ray_pos.y));
        const vec4 t2y = ray_inv_dir.y * (bmaxy - vec4(ray_pos.y));
        const vec4 miny = vec4::min(t1y, t2y);
        const vec4 maxy = vec4::max(t1y, t2y);

        const vec4 t1z = ray_inv_dir.z * (bminz - vec4(ray_pos.z));
        const vec4 t2z = ray_inv_dir.z * (bmaxz - vec4(ray_pos.z));
        const vec4 minz = vec4::min(t1z, t2z);
        const vec4 maxz = vec4::max(t1z, t2z);

        const vec4 tmin = vec4::max(minx, vec4::max(miny, minz));
        const vec4 tmax = vec4::min(maxx, vec4::min(maxy, maxz));

        // mask that tells which aabbs were hit
        const u32 hit = greater_than(tmin, vec4(0.0f)) & greater_equals(tmax, tmin);
        
        vec4 t(bits_to_f32((f32_to_bits(tmin[0]) & 0xfffffffc) | 0x0),
               bits_to_f32((f32_to_bits(tmin[1]) & 0xfffffffc) | 0x1),
               bits_to_f32((f32_to_bits(tmin[2]) & 0xfffffffc) | 0x2),
               bits_to_f32((f32_to_bits(tmin[3]) & 0xfffffffc) | 0x3));

        // sorting network
        sort_compare(t[0], t[1]);
        sort_compare(t[2], t[3]);
        sort_compare(t[0], t[2]);
        sort_compare(t[1], t[3]);
        sort_compare(t[2], t[3]);

        // remove the index from the t and add the bit to set if it was a hit or not
        t[0] = (f32_to_bits(t[0]) & 0xfffffffc) | (!!((1 << (f32_to_bits(t[0]) & 0x00000003)) & hit));
        t[1] = (f32_to_bits(t[1]) & 0xfffffffc) | (!!((1 << (f32_to_bits(t[1]) & 0x00000003)) & hit));
        t[2] = (f32_to_bits(t[2]) & 0xfffffffc) | (!!((1 << (f32_to_bits(t[2]) & 0x00000003)) & hit));
        t[3] = (f32_to_bits(t[3]) & 0xfffffffc) | (!!((1 << (f32_to_bits(t[3]) & 0x00000003)) & hit));

        return { t };
    }
};
#endif

struct alignas(16) AABB {
    alignas(16) vec3 bmin;
    alignas(16) vec3 bmax;

    // The union of two AABBs
    AABB merge(AABB other) const {
        return { { min(bmin.x, other.bmin.x), min(bmin.y, other.bmin.y), min(bmin.z, other.bmin.z) },
                 { max(bmax.x, other.bmax.x), max(bmax.y, other.bmax.y), max(bmax.z, other.bmax.z) } };
    }

    // Extend the AABB such that the given point is also contained
    AABB extend(vec3 p) const {
        return { { min(bmin.x, p.x), min(bmin.y, p.y), min(bmin.z, p.z) },
                 { max(bmax.x, p.x), max(bmax.y, p.y), max(bmax.z, p.z) } };
    }

    // The point in the middle of the AABB
    vec3 centroid() const {
        return 0.5f * bmin + 0.5f * bmax;
    }

    vec3 diagonal() const {
        return bmax - bmin;
    }

    f32 surface_area() const {
        const vec3 d = diagonal();
        return abs(2.0f * (d.x * d.y + d.x * d.z + d.y * d.z));
    }

    // Return the axis for which the AABB is largest
    u32 max_extend() const {
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

    __device__ bool intersect(vec3 ray_pos, vec3 ray_inv_dir, f32* t_out) const {
        const vec3 t1 = (bmin - ray_pos) * ray_inv_dir;
        const vec3 t2 = (bmax - ray_pos) * ray_inv_dir;

        const vec3 emin = vec3::min(t1, t2);
        const vec3 emax = vec3::max(t1, t2);

        const f32 tmin = max(emin.x, max(emin.y, emin.z));
        const f32 tmax = min(emax.x, min(emax.y, emax.z));

        *t_out = tmin;

        return tmax > 0.0f && tmax >= tmin;
    }

    static AABB empty() {
        return { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
    }

    __host__ static AABB for_triangle(vec3 p0, vec3 p1, vec3 p2) {
        return { { min({ p0.x, p1.x, p2.x }), min({ p0.y, p1.y, p2.y }), min({ p0.z, p1.z, p2.z }) },
                 { max({ p0.x, p1.x, p2.x }), max({ p0.y, p1.y, p2.y }), max({ p0.z, p1.z, p2.z }) } };
    }
};

#endif // GUARD_AABB_H
