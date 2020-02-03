#ifndef GUARD_VECMATH_H
#define GUARD_VECMATH_H

#define _USE_MATH_DEFINES // for PI constants
#include <math.h>

union vec3 {
    struct { float x, y, z; };
    struct { float r, g, b; };
    float fields[3];

    __host__ __device__ vec3() { }
    __host__ __device__ vec3(float x, float y, float z): x(x), y(y), z(z) { }

    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ vec3 normalize() const {
        const float l = length();
        return {x / l, y / l, z / l };
    }

    __host__ __device__ float operator[](size_t i) const {
        return fields[i];
    }
};

inline __host__ __device__ float dot(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ vec3 cross(vec3 a, vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

inline __host__ __device__ vec3 operator*(float s, vec3 v) {
    return { s * v.x, s * v.y, s * v.z };
}

inline __host__ __device__ vec3 operator+(vec3 a, vec3 b) {
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

inline __host__ __device__ vec3 operator-(vec3 a, vec3 b) {
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

union alignas(16) vec4 {
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
    uint32_t hit;
    vec3 normal;
    float t;

    __device__ static HitRecord no_hit() {
        HitRecord rec;
        rec.hit = 0;
        return rec;
    }
};

struct Triangle {
    vec3 v0, v1, v2;

    __device__ void intersect(Ray ray, HitRecord* record) const {
        record->hit = 0;

        const float EPSILON = 0.0001f;

        const vec3 e0 = v1 - v0;
        const vec3 e1 = v2 - v0;

        const vec3 h = cross(ray.dir, e1);
        const float a = dot(e0, h);

        if (a > (-EPSILON) && a < EPSILON) {
            return;
        }

        const float f = 1.0f / a;
        const vec3 s = ray.pos - v0;
        const float u = f * dot(s, h);
        if (u < 0.0f || u > 1.0f) {
            return;
        }

        const vec3 q = cross(s, e0);
        const float v = f * dot(ray.dir, q);
        if (v < 0.0f || (u + v) > 1.0f) {
            return;
        }

        const float t = f * dot(e1, q);

        if (t < EPSILON) {
            return;
        }

        vec3 n = cross(e0, e1).normalize();
        if (dot(n, ray.dir) > 0.0f) { n = -1.0f * n; }

        record->hit = 1;
        record->pos = ray.pos + t * ray.dir;
        record->normal = n;
        record->t = t;
    }
};

struct Sphere {
    vec3 pos;
    float radius;

    __device__ void intersect(Ray ray, HitRecord* record) const {
        record->hit = 0;

        const vec3 o = ray.pos - pos;
        const float b = dot(o, ray.dir);
        const float c = dot(o, o) - radius * radius;

        if (b > 0.0f && c > 0.0f) {
            return;
        }

        const float d = b * b - c;

        if (d < 0.0f) {
            return;
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
        if (t_far < 0.0f) { 
            return; 
        }

        const float t_closest = (t_near < 0.0f) ? t_far : t_near;

        const vec3 isect_on_sphere = ray.pos + t_closest * ray.dir;
        const vec3 n = (isect_on_sphere - pos).normalize();

        // NOTE
        // for now just report a true hit if at least one of the t is positive
        // this could also mean we are inside the sphere however.

        record->hit = 1;
        record->pos = isect_on_sphere;
        record->normal = n;
        record->t = t_closest;
    }
};

inline __host__ __device__ void cartesian_to_spherical(vec3 v, float* inclination, float* azimuth) {
    vec3 n = v.normalize();
    *inclination = acos(n.y);
    *azimuth = atan(n.z / n.x);
}

inline __host__ __device__ vec3 spherical_to_cartesian(float inclination, float azimuth) {
    return { sinf(inclination) * cosf(azimuth),
             cosf(inclination),
             sinf(inclination) * sinf(azimuth) };
}

template<typename T>
inline T clamp(T x, T a, T b) {
    return std::min(b, std::max(x, a));
}

struct PointCamera {
    vec3 pos;

    float inclination; // range [0, PI)
    float azimuth; // range [0, 2*PI)

    float width, height, plane_distance;

    vec3 u, v, w;

    PointCamera(vec3 pos, vec3 up, vec3 at, float width, float height, float plane_distance) {
        const vec3 forward = (at - pos).normalize();

        cartesian_to_spherical(forward, &inclination, &azimuth);

        // This approach may be a bit naive, not sure
        w = forward;
        u = cross(w, up);
        v = cross(w, u);

        this->pos = pos;
        this->width = width;
        this->height = height;
        this->plane_distance = plane_distance;
    }

    // NOTE
    // This is at least a usable camera, but I feel like it's a bit weird when
    // close to looking straight up or down. Also maybe initialize with an fov
    // instead of a plane distance, that way it is a bit more intuitive.
    void update_uvw() {
        // make sure that inclination and azimuth are in a valid range
        inclination = clamp<float>(inclination, 0.0001f, M_PI - 0.0001f);

        //if (azimuth >= M_2_PI) { azimuth -= M_2_PI; }
        //if (azimuth < 0.0f) { azimuth += M_2_PI; }

        w = spherical_to_cartesian(inclination, azimuth);
        u = cross(w, vec3(0.0f, 1.0f, 0.0f));
        v = cross(w, u);
    }

    // u and v are in range (-1.0f, 1.0f)
    __device__ Ray create_ray(float x, float y) { 
        const float px = 0.5f * x * width;
        const float py = 0.5f * y * height;

        const vec3 p = plane_distance * w + px * u + py * v;

        return { pos, p.normalize() };
    }
};

#endif // GUARD_VECMATH_H
