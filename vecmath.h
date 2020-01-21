#ifndef GUARD_VECMATH_H
#define GUARD_VECMATH_H

union vec3 {
    struct { float x, y, z; };
    struct { float r, g, b; };
    float fields[3];

    __device__ float length() {
        return sqrtf(x*x + y*y + z*z);
    }

    __device__ vec3 normalize() {
        const float l = length();
        return {x / l, y / l, z / l };
    }
};

inline __device__ float dot(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ vec3 operator*(float s, vec3 v) {
    return { s * v.x, s * v.y, s * v.z };
}

inline __device__ vec3 operator+(vec3 a, vec3 b) {
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

inline __device__ vec3 operator-(vec3 a, vec3 b) {
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

union vec4 {
    struct { float x, y, z, w; };
    struct { float r, g, b, a; };
    float fields[4];
};

struct Ray {
    vec3 pos;
    vec3 dir;
};

struct HitRecord {
    vec3 pos;
    vec3 normal;
    float t;
};

struct Sphere {
    vec3 pos;
    float radius;

    __device__ uint32_t intersect(Ray ray, HitRecord* record) {
        const vec3 o = ray.pos - pos;
        const float b = dot(o, ray.dir);
        const float c = dot(o, o) - radius * radius;

        if (b > 0.0f && c > 0.0f) {
            return 0;
        }

        const float d = b * b - c;

        if (d < 0.0f) {
            return 0;
        }

        const float ds = sqrtf(d);

        const float t0 = -b - ds;
        const float t1 = -b + ds;

        float t_near, t_far;
        if (t0 > t1) { 
            t_near = t1; 
            t_far = t0; 
        } else { 
            t_near = t0; 
            t_far = t1; 
        }

        // If t_far is smaller than 0, then t_near is also smaller than 0 so
        // both are negative which means that the sphere intersection is behind
        // us.
        if (t_far < 0.0f) { return 0; }

        const float t_closest = (t_near < 0.0f) ? t_far : t_near;

        const vec3 isect_on_sphere = ray.pos + t_closest * ray.dir;
        const vec3 n = (isect_on_sphere - pos).normalize();

        record->t = t_closest;
        record->pos = pos;
        record->normal = n;

        // NOTE
        // for now just report a true hit if at least one of the t is positive
        // this could also mean we are inside the sphere however.

        return 1;
    }
};

inline __device__ void cartesian_to_spherical(vec3 v, float* inclination, float* azimuth) {
    vec3 n = v.normalize();
    *inclination = acos(n.y);
    *azimuth = atan(n.z / n.x);
}

inline __device__ vec3 spherical_to_cartesian(float inclination, float azimuth) {
    return { sinf(inclination) * cosf(azimuth),
             cosf(inclination),
             sinf(inclination) * sinf(azimuth) };
}

#endif // GUARD_VECMATH_H
