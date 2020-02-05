#ifndef GUARD_VECMATH_H
#define GUARD_VECMATH_H

#define _USE_MATH_DEFINES // for PI constants
#include <math.h>
#include <algorithm>

union vec3 {
    struct { float x, y, z; };
    struct { float r, g, b; };
    float fields[3];

    __host__ __device__ vec3() { }
    __host__ __device__ vec3(float a): x(a), y(a), z(a) { }
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

    __host__ __device__ vec3 inverse() const {
        return { 1.0f / x, 1.0f / y, 1.0f / z };
    }
};

template<typename T>
__device__ const T& min(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template<typename T>
__device__ const T& max(const T& a, const T& b) {
    return (a > b) ? a : b;
}

template<typename T>
__device__ constexpr T min_recurse(const T* begin, const T* end) {
    // I would rather have this be an if/else but a constexpr function may only
    // have one return statement
    return (begin + 1 == end) ? (*begin) : min(*begin, min_recurse(begin + 1, end));
}

template<typename T>
__device__ constexpr T min(std::initializer_list<T> l) {
    return min_recurse(l.begin(), l.end());
}

template<typename T>
__device__ constexpr T max_recurse(const T* begin, const T* end) {
    return (begin + 1 == end) ? (*begin) : max(*begin, max_recurse(begin + 1, end));
}

template<typename T>
__device__ constexpr T max(std::initializer_list<T> l) {
    return max_recurse(l.begin(), l.end());
}

// For each element select the minimum of the two vectors
inline __device__ vec3 min_elements(vec3 a, vec3 b) {
    return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
}

// For each element select the maximum of the two vectors
inline __device__ vec3 max_elements(vec3 a, vec3 b) {
    return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
}

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

inline __host__ __device__ vec3 operator*(vec3 a, vec3 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
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

    __host__ __device__ vec4(float v) : x(v), y(v), z(v), w(v) { }
    __host__ __device__ vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) { }
};

struct Ray {
    vec3 pos;
    vec3 dir;
};

struct HitRecord {
    vec3 pos;
    float t;
    vec3 normal;
};

inline __device__ vec3 triangle_normal(vec3 v0, vec3 v1, vec3 v2) {
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    return cross(e0, e1).normalize();
}

inline __device__ bool triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, float* t_out) {
    const float EPSILON = 0.000001f;

    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    const vec3 h = cross(ray.dir, e1);
    const float a = dot(e0, h);

    if (a > (-EPSILON) && a < EPSILON) {
        return false;
    }

    const float f = 1.0f / a;
    const vec3 s = ray.pos - v0;
    const float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const vec3 q = cross(s, e0);
    const float v = f * dot(ray.dir, q);
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    const float t = f * dot(e1, q);

    if (t < EPSILON) {
        return false;
    }

    *t_out = t;

    return true;
}

inline bool sphere_intersect(Ray ray, vec3 center, float radius, float* t_out) {
    const vec3 o = ray.pos - center;
    const float b = dot(o, ray.dir);
    const float c = dot(o, o) - radius * radius;

    if (b > 0.0f && c > 0.0f) {
        return false;
    }

    const float d = b * b - c;

    if (d < 0.0f) {
        return false;
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
        return false;
    }

    const float t_closest = (t_near < 0.0f) ? t_far : t_near;

    // NOTE
    // for now just report a true hit if at least one of the t is positive
    // this could also mean we are inside the sphere however.

    *t_out = t_closest;

    return true;
}

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

    // uu and vv are in range (-1.0f, 1.0f)
    __device__ Ray create_ray(float uu, float vv) { 
        const float px = 0.5f * uu * width;
        const float py = 0.5f * vv * height;

        const vec3 p = plane_distance * w + px * u + py * v;

        return { pos, p.normalize() };
    }
};

#endif // GUARD_VECMATH_H
