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

         __device__ inline vec3 min_elements(vec3 a, vec3 b);
         __device__ inline vec3 max_elements(vec3 a, vec3 b);
__host__ __device__ inline f32  dot(vec3 a, vec3 b);
__host__ __device__ inline vec3 cross(vec3 a, vec3 b);
__host__ __device__ inline vec3 operator*(f32 s, vec3 v);
__host__ __device__ inline vec3 operator*(vec3 a, vec3 b);
__host__ __device__ inline void operator*=(vec3& a, const vec3& b);
__host__ __device__ inline vec3 operator+(vec3 a, vec3 b);
__host__ __device__ inline vec3 operator-(vec3 a, vec3 b);
__host__ __device__ inline void operator+=(vec3& a, const vec3& b);

union alignas(16) vec4 {
    struct { float x, y, z, w; };
    struct { float r, g, b, a; };
    float fields[4];

    __host__ __device__ vec4() { }
    __host__ __device__ vec4(float v) : x(v), y(v), z(v), w(v) { }
    __host__ __device__ vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) { }
    __host__ __device__ vec4(const vec3& v, f32 w) : x(v.x), y(v.y), z(v.z), w(w) { }
};

__device__ inline vec4 operator*(f32 s, vec4 v);
__device__ inline vec4 operator+(const vec4&a, const vec4& b);
__device__ inline void operator+=(vec4& a, const vec4& b);
__device__ inline vec4 operator/(const vec4& a, f32 s);

struct alignas(16) Ray {
    vec3 pos; uint32_t pad0;
    vec3 dir; uint32_t pad1;

    __device__ Ray(vec3 pos, vec3 dir) : pos(pos), dir(dir) { }
};

// Random number generation
// ----------------------------------------------------------------------------

__device__ inline
f32 rng_xor32(u32& seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed <<  5;
    return seed * 2.3283064365387e-10f;
}

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
void cartesian_to_spherical(const vec3& v, float* inclination, float* azimuth) {
    const f32 r = v.length();
    *inclination = acosf(v.y / r); // in range [0, pi]
    *azimuth = atan2f(v.z, v.x); // in range [-pi, pi]
}

// inclination and azimuth in radians
inline __host__ __device__ 
vec3 spherical_to_cartesian(float inclination, float azimuth) {
    return { sinf(inclination) * cosf(azimuth),
             cosf(inclination),
             sinf(inclination) * sinf(azimuth) };
}

__device__ inline
void build_orthonormal_basis(const vec3& n, vec3* t, vec3* b) {
#if 1
    // This method can produce weird results along the planes splitting in
    // x / y if the resolution of a triangle mesh isn't high enough
    *t = (fabs(n.x) > fabs(n.y)) ? vec3(n.z, 0.0f, -n.x).normalize() : 
                                   vec3(0.0f, -n.z, n.y).normalize();
    *b = cross(n, *t);
#else
    // This method has some strange stuff going on near n = (0, 0, -1)
    // Frisvad method
    //if (n.z < -0.9999999f) {
    if (n.z == -1.0f) {
        *t = vec3( 0.0f, -1.0f, 0.0f);
        *b = vec3(-1.0f,  0.0f, 0.0f);
    } else {
        const f32 r = 1.0f / (1.0f + n.z);
        const f32 s = -n.x * n.y * r;
        *t = vec3(1.0f - n.x * n.x * r, s, -n.x);
        *b = vec3(s, 1.0f - n.y * n.y * r, -n.y);
    }
#endif
}

__device__ inline
vec3 to_world_space(const vec3& sample, const vec3& n, const vec3& t, const vec3& b) {
    return sample.x * b + sample.y * n + sample.z * t;
}

__device__ inline
vec3 reflect(const vec3& n, const vec3& incident) {
    return (incident - 2.0f * dot(n, incident) * n).normalize();
}

__device__ inline
vec3 sample_uniform_hemisphere(f32 r1, f32 r2) {
    const f32 s = sqrtf(1.0f - r1 * r1);
    const f32 phi = 2.0f * M_PI * r2; // [0, 2*pi)
    const f32 x = s * cosf(phi);
    const f32 z = s * sinf(phi);
    return vec3(x, r1, z);
}

__device__ inline
vec3 sample_cosine_weighted_hemisphere(f32 r1, f32 r2) {
    const f32 s = sqrtf(r1);
    const f32 theta = 2.0f * M_PI * r2;
    const f32 x = s * cosf(theta);
    const f32 z = s * sinf(theta);
    return vec3(x, sqrtf(1.0f - r1), z);
}

// vec3 functions
// ----------------------------------------------------------------------------

__host__ __device__ inline float vec3::length() const { return sqrtf(x*x + y*y + z*z); }
__host__ __device__ inline vec3  vec3::normalize() const { const float l = length(); return {x / l, y / l, z / l }; }
__host__ __device__ inline vec3  vec3::inverse() const { return { 1.0f / x, 1.0f / y, 1.0f / z }; }
__host__ __device__ inline float vec3::operator[](size_t i) const { return fields[i]; }

// For each element select the minimum of the two vectors
__device__ inline
vec3 min_elements(vec3 a, vec3 b) {
    return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
}

// For each element select the maximum of the two vectors
__device__ inline
vec3 max_elements(vec3 a, vec3 b) {
    return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
}

__host__ __device__ inline
float dot(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline 
vec3 cross(vec3 a, vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x, };
}

__host__ __device__ inline
vec3 operator*(float s, vec3 v) {
    return { s * v.x, s * v.y, s * v.z };
}

__host__ __device__ inline
vec3 operator*(vec3 a, vec3 b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__host__ __device__ inline 
void operator*=(vec3& a, const vec3& b) {
    a = a * b;
}

__host__ __device__ inline
vec3 operator+(vec3 a, vec3 b) {
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

__host__ __device__ inline 
vec3 operator-(vec3 a, vec3 b) {
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

__host__ __device__ inline 
void operator+=(vec3& a, const vec3& b) {
    a = a + b;
}

// vec4
// ----------------------------------------------------------------------------

__device__ inline
vec4 operator*(f32 s, vec4 v) {
    return { s * v.x, s * v.y, s * v.z, s * v.w };
}
__device__ inline 
vec4 operator+(const vec4&a, const vec4& b) {
    return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ inline 
void operator+=(vec4& a, const vec4& b) {
    a = a + b;
}

__device__ inline 
vec4 operator/(const vec4& a, f32 s) {
    return vec4(a.x / s, a.y / s, a.z / s, a.w / s);
}

// Normal packing
// ----------------------------------------------------------------------------

// For theory see http://aras-p.info/texts/CompactNormalStorage.html
// This version was humbly copied from Lighthouse 2

inline uint32_t pack_normal(vec3 n) {
    const float f = 65535.0f / sqrtf(8.0f * n.z + 8.0f);
    return cast(uint32_t, n.x * f + 32767.0f) + (cast(uint32_t, n.y * f + 32767.0f ) << 16);
}

inline vec3 unpack_normal(uint32_t p) {
    vec4 nn(cast(float, p & 65335) * (2.0f / 65335.0f) - 1.0f,
            cast(float, p >> 16) * (2.0f / 65335.0f) - 1.0f,
            -1.0f,
            -1.0f);

    const float l = dot(vec3(nn.x, nn.y, nn.z), vec3(-nn.x, -nn.y, -nn.w));
    nn.z = l;

    const float ll = sqrtf(l);
    nn.x *= ll;
    nn.y *= ll;

    return vec3(nn.x, nn.y, nn.z) * 2.0f + vec3(0.0f, 0.0f, -1.0f);
}

// triangle functions
// ----------------------------------------------------------------------------

__host__ __device__ inline
vec3 triangle_normal(vec3 v0, vec3 v1, vec3 v2) {
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    return cross(e0, e1).normalize();
}

__device__ inline
vec3 triangle_normal_lerp(vec3 n0, vec3 n1, vec3 n2, float u, float v) {
    const float w = 1.0f - u - v;
    return w * n0 + u * n1 + v * n2;
}

// Moller Trumbore ray triangle intersection algorithm
// This algorithm calculates the Barycentric coordinates to check for
// intersection, so we also return those.
__device__ inline
bool triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, 
                        float* t_out, float* u_out, float* v_out)
{
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
    *u_out = u;
    *v_out = v;

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
