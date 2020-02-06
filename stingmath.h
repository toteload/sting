#ifndef GUARD_STINGMATH_H
#define GUARD_STINGMATH_H

#define _USE_MATH_DEFINES // for PI constants
#include <math.h>
#include <initializer_list>

// This value has been empirically found, there is a good chance another value is better :)
__device__ const float TRIANGLE_INTERSECT_EPSILON = 1e-6f;

union vec3 {
    struct { float x, y, z; };
    struct { float r, g, b; };
    float fields[3];

    __host__ __device__ vec3() { }
    __host__ __device__ vec3(float a): x(a), y(a), z(a) { }
    __host__ __device__ vec3(float x, float y, float z): x(x), y(y), z(z) { }

    __host__ __device__ inline float length() const;
    __host__ __device__ inline vec3  normalize() const;
    __host__ __device__ inline vec3  inverse() const;
    __host__ __device__ inline float operator[](size_t i) const;
};

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

// General functions
// ----------------------------------------------------------------------------

template<typename T>
__host__ __device__ 
const T& min(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template<typename T>
__host__ __device__ 
const T& max(const T& a, const T& b) {
    return (a > b) ? a : b;
}

template<typename T>
__host__ __device__ constexpr 
T min_recurse(const T* begin, const T* end) {
    // I would rather have this be an if/else but a constexpr function may only
    // have one return statement
    return (begin + 1 == end) ? (*begin) : min(*begin, min_recurse(begin + 1, end));
}

template<typename T>
__host__ __device__ constexpr 
T min(std::initializer_list<T> l) {
    return min_recurse(l.begin(), l.end());
}

template<typename T>
__host__ __device__ constexpr 
T max_recurse(const T* begin, const T* end) {
    return (begin + 1 == end) ? (*begin) : max(*begin, max_recurse(begin + 1, end));
}

template<typename T>
__host__ __device__ constexpr 
T max(std::initializer_list<T> l) {
    return max_recurse(l.begin(), l.end());
}

template<typename T>
inline T clamp(T x, T a, T b) {
    return min(b, max(x, a));
}

inline float degrees_to_radians(float d) {
    return (d / 180.0f) * M_PI;
}

// inclination and azimuth in radians
inline __host__ __device__ 
void cartesian_to_spherical(vec3 v, float* inclination, float* azimuth) {
    vec3 n = v.normalize();
    *inclination = acos(n.y);
    *azimuth = atanf(n.z / n.x);
}

// inclination and azimuth in radians
inline __host__ __device__ 
vec3 spherical_to_cartesian(float inclination, float azimuth) {
    return { sinf(inclination) * cosf(azimuth),
             cosf(inclination),
             sinf(inclination) * sinf(azimuth) };
}

// vec3 functions
// ----------------------------------------------------------------------------

__host__ __device__ inline float vec3::length() const { return sqrtf(x*x + y*y + z*z); }
__host__ __device__ inline vec3  vec3::normalize() const { const float l = length(); return {x / l, y / l, z / l }; }
__host__ __device__ inline vec3  vec3::inverse() const { return { 1.0f / x, 1.0f / y, 1.0f / z }; }
__host__ __device__ inline float vec3::operator[](size_t i) const { return fields[i]; }

// For each element select the minimum of the two vectors
inline __device__ 
vec3 min_elements(vec3 a, vec3 b) {
    return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
}

// For each element select the maximum of the two vectors
inline __device__ 
vec3 max_elements(vec3 a, vec3 b) {
    return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
}

inline __host__ __device__ 
float dot(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ 
vec3 cross(vec3 a, vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x, };
}

inline __host__ __device__ 
vec3 operator*(float s, vec3 v) {
    return { s * v.x, s * v.y, s * v.z };
}

inline __host__ __device__ 
vec3 operator*(vec3 a, vec3 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

inline __host__ __device__ 
vec3 operator+(vec3 a, vec3 b) {
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

inline __host__ __device__ 
vec3 operator-(vec3 a, vec3 b) {
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

// triangle functions
// ----------------------------------------------------------------------------

inline __device__ vec3 triangle_normal(vec3 v0, vec3 v1, vec3 v2) {
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    return cross(e0, e1).normalize();
}

inline __device__ bool triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, float* t_out) {
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    const vec3 h = cross(ray.dir, e1);
    const float a = dot(e0, h);

    if (a > (-TRIANGLE_INTERSECT_EPSILON) && a < TRIANGLE_INTERSECT_EPSILON) {
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

    if (t < TRIANGLE_INTERSECT_EPSILON) {
        return false;
    }

    *t_out = t;

    return true;
}

// Sphere intersect, obviously but I wanted to separate it from the triangle
// functions above :)
// ----------------------------------------------------------------------------

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

#endif // GUARD_STINGMATH_H
