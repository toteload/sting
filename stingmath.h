#ifndef GUARD_STINGMATH_H
#define GUARD_STINGMATH_H

#include "dab/dab.h"

#define _USE_MATH_DEFINES // for PI constants
#include <math.h>
#include <float.h>
#include <initializer_list>

#if 0
union vec4;

union Vector3 {
    struct { f32 x, y, z; };
    struct { f32 r, g, b; };
    f32 fields[3];

    __host__ __device__ Vector3() { }
    __host__ __device__ explicit Vector3(f32 a): x(a), y(a), z(a) { }
    __host__ __device__ Vector3(f32 x, f32 y, f32 z): x(x), y(y), z(z) { }
    __host__ __device__ Vector3(const vec4& v);

    __host__ __device__ inline f32  length_squared() const;
    __host__ __device__ inline f32  length() const;
    __host__ __device__ inline Vector3 normalize() const;
    __host__ __device__ inline Vector3 inverse() const;
    __host__ __device__ inline f32  operator[](size_t i) const;

    __host__ __device__ inline static Vector3 min(const Vector3& a, const Vector3& b);
    __host__ __device__ inline static Vector3 max(const Vector3& a, const Vector3& b);
};

__host__ __device__ inline f32  dot(const Vector3& a, const Vector3& b);
__host__ __device__ inline Vector3 cross(const Vector3& a, const Vector3& b);
__host__ __device__ inline Vector3 operator*(f32 s, const Vector3& v);
__host__ __device__ inline Vector3 operator*(const Vector3& v, f32 s);
__host__ __device__ inline Vector3 operator*(const Vector3& a, const Vector3& b);
__host__ __device__ inline void operator*=(Vector3& a, const Vector3& b);
__host__ __device__ inline Vector3 operator+(const Vector3& a, const Vector3& b);
__host__ __device__ inline Vector3 operator-(const Vector3& a, const Vector3& b);
__host__ __device__ inline void operator+=(Vector3& a, const Vector3& b);

union alignas(16) vec4 {
    struct { f32 x, y, z, w; };
    struct { f32 r, g, b, a; };
    f32 fields[4];

    __host__ __device__ vec4() { }
    __host__ __device__ explicit vec4(f32 v) : x(v), y(v), z(v), w(v) { }
    __host__ __device__ vec4(f32 x, f32 y, f32 z, f32 w) : x(x), y(y), z(z), w(w) { }
    __host__ __device__ vec4(const Vector3& v, f32 w) : x(v.x), y(v.y), z(v.z), w(w) { }

    __host__ __device__ inline f32& operator[](size_t i) const;

    __host__ __device__ inline static vec4 min(const vec4& a, const vec4& b);
    __host__ __device__ inline static vec4 max(const vec4& a, const vec4& b);
};

__device__ inline vec4 operator*(f32 s, const vec4& v);
__device__ inline vec4 operator-(const vec4& a, const vec4& b);
__device__ inline vec4 operator+(const vec4& a, const vec4& b);
__device__ inline void operator+=(vec4& a, const vec4& b);
__device__ inline vec4 operator/(const vec4& a, f32 s);
__device__ inline u32 greater_than(const vec4& a, const vec4& b);
__device__ inline u32 greater_equals(const vec4& a, const vec4& b);
#endif

struct alignas(16) Ray {
    Vector3 pos; f32 tmin;
    Vector3 dir; f32 tmax;

    __device__ Ray(Vector3 pos, Vector3 dir) : pos(pos), tmin(FLT_MIN), dir(dir), tmax(FLT_MAX) { }
    __device__ Ray(Vector3 pos, Vector3 dir, f32 tmin, f32 tmax) : pos(pos), tmin(tmin), dir(dir), tmax(tmax) { }
};

// General functions
// ----------------------------------------------------------------------------

#if 0
template<typename T>
__host__ __device__ 
T min(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template<typename T>
__host__ __device__ 
T max(const T& a, const T& b) {
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
#endif

union Float32 {
    f32 f;
    u32 u;

    explicit Float32(f32 x) : f(x) { }
    explicit Float32(u32 x) : u(x) { }

    // NOTE: The arguments here are the bit values not the actual values the
    // represent so sign is not -1 or 1, but 0 or 1.
    explicit Float32(u32 sign, u32 exponent, u32 mantissa) :
        mantissa(mantissa), exponent(exponent), sign(sign)
    { }

    struct {
        u32 mantissa : 23;
        u32 exponent : 8;
        u32 sign : 1;
    };
};

inline u32 f32_to_bits(f32 x) {
    return Float32(x).u;
}

inline f32 bits_to_f32(u32 x) {
    return Float32(x).f;
}

// after calling this a will be min(a, b) and b max(a, b)
template<typename T>
void sort_compare(T& a, T& b) {
    if (a > b) { std::swap(a, b); }
}

inline f32 degrees_to_radians(f32 d) {
    return (d / 180.0f) * M_PI;
}

// inclination and azimuth in radians
__device__ inline
void cartesian_to_spherical(const Vector3& v, f32* inclination, f32* azimuth) {
    const f32 r = v.length();
    *inclination = acosf(v.y / r); // in range [0, pi]
    *azimuth = atan2f(v.z, v.x); // in range [-pi, pi]
}

// inclination and azimuth in radians
inline 
Vector3 spherical_to_cartesian(f32 inclination, f32 azimuth) {
    return vec3(sinf(inclination) * cosf(azimuth),
                cosf(inclination),
                sinf(inclination) * sinf(azimuth));
}

__device__ inline
void build_orthonormal_basis(const Vector3& n, Vector3* t, Vector3* b) {
    *t = (fabs(n.x) > fabs(n.y)) ? vec3(n.z, 0.0f, -n.x).normalized() : 
                                   vec3(0.0f, -n.z, n.y).normalized();
    *b = cross(n, *t);
}

__device__ inline
Vector3 to_world_space(const Vector3& sample, const Vector3& n, const Vector3& t, const Vector3& b) {
    return sample.x * b + sample.y * n + sample.z * t;
}

__device__ inline
Vector3 reflect(const Vector3& n, const Vector3& incident) {
    return (incident - 2.0f * dot(n, incident) * n).normalized();
}

__device__ inline
Vector3 sample_uniform_hemisphere(f32 r1, f32 r2) {
    const f32 s = sqrtf(1.0f - r1 * r1);
    const f32 phi = 2.0f * M_PI * r2; // [0, 2*pi)
    const f32 x = s * cosf(phi);
    const f32 z = s * sinf(phi);
    return vec3(x, r1, z);
}

__device__ inline
Vector3 sample_cosine_weighted_hemisphere(f32 r1, f32 r2) {
    const f32 s = sqrtf(r1);
    const f32 theta = 2.0f * M_PI * r2;
    const f32 x = s * cosf(theta);
    const f32 z = s * sinf(theta);
    return vec3(x, sqrtf(1.0f - r1), z);
}

__device__ inline
f32 balance_heuristic(i32 nf, f32 f_pdf, i32 ng, f32 g_pdf) {
    return (nf * f_pdf) / (nf * f_pdf + ng * g_pdf);
}

__device__ inline
f32 power_heuristic(i32 nf, f32 f_pdf, i32 ng, f32 g_pdf) {
    const f32 f = nf * f_pdf;
    const f32 g = ng * g_pdf;
    return (f * f) / (f * f + g * g);
}

// Vector3 functions
// ----------------------------------------------------------------------------

#if 0
__host__ __device__ inline Vector3::Vector3(const vec4& v) : x(v.x), y(v.y), z(v.z) { }
__host__ __device__ inline f32  Vector3::length_squared() const { return x*x + y*y + z*z; }
__host__ __device__ inline f32  Vector3::length() const { return sqrtf(x*x + y*y + z*z); }
__host__ __device__ inline Vector3 Vector3::normalize() const { const f32 l = length(); return {x / l, y / l, z / l }; }
__host__ __device__ inline Vector3 Vector3::inverse() const { return { 1.0f / x, 1.0f / y, 1.0f / z }; }
__host__ __device__ inline f32  Vector3::operator[](size_t i) const { return fields[i]; }

// For each element select the minimum of the two vectors
__host__ __device__ inline
Vector3 Vector3::min(const Vector3& a, const Vector3& b) {
    return { ::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z) };
}

// For each element select the maximum of the two vectors
__host__ __device__ inline
Vector3 Vector3::max(const Vector3& a, const Vector3& b) {
    return { ::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z) };
}

__host__ __device__ inline
f32 dot(const Vector3& a, const Vector3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline 
Vector3 cross(const Vector3& a, const Vector3& b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x, };
}

__host__ __device__ inline
Vector3 operator*(f32 s, const Vector3& v) {
    return { s * v.x, s * v.y, s * v.z };
}
__host__ __device__ inline 
Vector3 operator*(const Vector3& v, f32 s) {
    return s * v;
}

__host__ __device__ inline
Vector3 operator*(const Vector3& a, const Vector3& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__host__ __device__ inline 
void operator*=(Vector3& a, const Vector3& b) {
    a = a * b;
}

__host__ __device__ inline
Vector3 operator+(const Vector3& a, const Vector3& b) {
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

__host__ __device__ inline 
Vector3 operator-(const Vector3& a, const Vector3& b) {
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

__host__ __device__ inline 
void operator+=(Vector3& a, const Vector3& b) {
    a = a + b;
}
#endif

// vec4
// ----------------------------------------------------------------------------

#if 0
__host__ __device__ inline
f32& vec4::operator[](size_t i) const {
    return const_cast<f32&>(fields[i]);
}

__host__ __device__ inline
vec4 vec4::min(const vec4& a, const vec4& b) {
    return { ::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z), ::min(a.w, b.w) };
}

__host__ __device__ inline
vec4 vec4::max(const vec4& a, const vec4& b) {
    return { ::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z), ::max(a.w, b.w) };
}

__device__ inline
vec4 operator*(f32 s, const vec4& v) {
    return { s * v.x, s * v.y, s * v.z, s * v.w };
}

__device__ inline
vec4 operator-(const vec4& a, const vec4& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

__device__ inline 
vec4 operator+(const vec4& a, const vec4& b) {
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

__device__ inline
u32 greater_than(const vec4& a, const vec4& b) {
    return (cast(u32, a[3] > b[3]) << 3) |
           (cast(u32, a[2] > b[2]) << 2) |
           (cast(u32, a[1] > b[1]) << 1) |
           (cast(u32, a[0] > b[0])     );
}

__device__ inline
u32 greater_equals(const vec4& a, const vec4& b) {
    return (cast(u32, a[3] >= b[3]) << 3) |
           (cast(u32, a[2] >= b[2]) << 2) |
           (cast(u32, a[1] >= b[1]) << 1) |
           (cast(u32, a[0] >= b[0])     );
}
#endif

// Normal packing
// ----------------------------------------------------------------------------

__device__ inline
u32 pack_normal(Vector3 n) {
    const i16 nx = cast(i16, n.x * 32767.0f);
    const i16 nz = cast(i16, n.z * 32767.0f);
    return (cast(u32, nx) << 16) | (cast(u32, nz) & 0xffff);
}

__device__ inline 
Vector3 unpack_normal(u32 pn) {
    const f32 nx = (pn >> 16) / 32767.0f;
    const f32 nz = (pn & 0xffff) / 32767.0f;
    const f32 ny = sqrtf(1.0f - nx*nx - nz*nz);
    return vec3(nx, ny, nz);
}

// For theory see http://aras-p.info/texts/CompactNormalStorage.html
// This version was humbly copied from Lighthouse 2

// This doesn't work (yet)
#if 0
inline u32 pack_normal(Vector3 n) {
    const f32 f = 65535.0f / max(sqrtf(8.0f * n.z + 8.0f), 0.0001f);
    return cast(u32, n.x * f + 32767.0f) + (cast(u32, n.y * f + 32767.0f) << 16);
}

inline __device__ Vector3 unpack_normal(u32 p) {
    vec4 nn(cast(f32, p & 65335) * (2.0f / 65335.0f) - 1.0f,
            cast(f32, p >> 16) * (2.0f / 65335.0f) - 1.0f,
            -1.0f,
            -1.0f);

    const f32 l = dot(Vector3(nn.x, nn.y, nn.z), Vector3(-nn.x, -nn.y, -nn.w));
    nn.z = l;

    const f32 ll = sqrtf(l);
    nn.x *= ll;
    nn.y *= ll;

    return Vector3(nn.x, nn.y, nn.z) * 2.0f + Vector3(0.0f, 0.0f, -1.0f);
}
#endif

// triangle functions
// ----------------------------------------------------------------------------

__device__ inline
f32 triangle_solid_angle(const Vector3& a, const Vector3& b, const Vector3& c) {
    const f32 da = a.length();
    const f32 db = b.length();
    const f32 dc = c.length();

    return 2.0f * atan2f(dot(a, cross(b, c)), da*db*dc + dot(a, b)*dc + dot(a, c)*db + dot(b, c)*da);
}

__device__ inline
f32 triangle_solid_angle_hemisphere(const Vector3& a, const Vector3& b, const Vector3& c) {
    const u32 a_below = a.y < 0;
    const u32 b_below = b.y < 0;
    const u32 c_below = c.y < 0;

    const u32 below_hemisphere = a_below + b_below + c_below;

    switch (below_hemisphere) {
    case 3: { return 0.0f; }
    case 2: { 
        // two points of the triangle are below the hemisphere so the part of
        // the triangle that is in the hemisphere is shaped like a triangle
        Vector3 top, p0, p1;
        if (!a_below) { top = a; p0 = b; p1 = c; }
        if (!b_below) { top = b; p0 = c; p1 = a; }
        if (!c_below) { top = c; p0 = a; p1 = b; }

        const Vector3 e0 = (p0 - top);
        const Vector3 e1 = (p1 - top);

        const Vector3 bb = (-top.y / e0.y) * e0;
        const Vector3 cc = (-top.y / e1.y) * e1;

        return fabs(triangle_solid_angle(top, bb, cc));
    } break;
    case 1: { 
        // one point is below the hemisphere so the visible part is a 4-gon.
        // We need to do 2 triangle solid angle calculations.
        // Here we calculate the solid angle of the whole triangle and then
        // subtract the solid angle of the triangular part that is below the
        // hemisphere.
        // You can also divide the 4-gon into two triangles and then sum
        // their areas. This might be a bit faster, because there are some
        // overlapping calculations then.

        Vector3 bot, p0, p1;
        if (a_below) { bot = a; p0 = b; p1 = c; }
        if (b_below) { bot = b; p0 = c; p1 = a; }
        if (c_below) { bot = c; p0 = a; p1 = b; }

        const Vector3 e0 = (p0 - bot);
        const Vector3 e1 = (p1 - bot);

        const Vector3 bb = (-bot.y / e0.y) * e0;
        const Vector3 cc = (-bot.y / e1.y) * e1;

        return fabs(triangle_solid_angle(a, b, c)) - fabs(triangle_solid_angle(bot, bb, cc));
    } break;
    case 0: { return triangle_solid_angle(a, b, c); }

    // Just to shut up the compiler
    default: { return 0; }
    } 
}

__device__ inline
Vector3 triangle_normal(Vector3 v0, Vector3 v1, Vector3 v2) {
    const Vector3 e0 = v1 - v0;
    const Vector3 e1 = v2 - v0;

    return cross(e0, e1).normalized();
}

__device__ inline
Vector3 triangle_normal_lerp(Vector3 n0, Vector3 n1, Vector3 n2, f32 u, f32 v) {
    const f32 w = 1.0f - u - v;
    return w * n0 + u * n1 + v * n2;
}

__device__ inline
f32 triangle_area(const Vector3& p0, const Vector3& p1, const Vector3& p2) {
    const f32 a = (p0 - p1).length();
    const f32 b = (p1 - p2).length();
    const f32 c = (p2 - p0).length();
    const f32 s = (a + b + c) / 2.0f;
    return sqrtf(s * (s - a) * (s - b) * (s - c));
}

__device__ inline
Vector3 triangle_random_point(const Vector3& p0, const Vector3& p1, const Vector3& p2, f32 r1, f32 r2) {
    if (r1 + r2 > 1.0f ) {
        r1 = 1.0f - r1;
        r2 = 1.0f - r2;
    }

    const f32 w = 1.0f - r1 - r2;
    return r1 * p0 + r2 * p1 + w * p2;
}

struct alignas(16) TriangleIntersection {
    f32 t, u, v; u32 hit;

    __device__ static TriangleIntersection no_hit() { 
        TriangleIntersection isect;
        isect.t = isect.u = isect.v = 0.0f;
        isect.hit = 0;
        return isect;
    }
};

// Moller Trumbore ray triangle intersection algorithm
// This algorithm calculates the Barycentric coordinates to check for
// intersection, so we also return those.
__device__ inline
TriangleIntersection triangle_intersect(Ray ray, Vector3 v0, Vector3 v1, Vector3 v2) {
    // This value has been empirically found, there is a good chance another value is better :)
    constexpr f32 TRIANGLE_INTERSECT_EPSILON = 1e-6f;

    const Vector3 e0 = v1 - v0;
    const Vector3 e1 = v2 - v0;

    const Vector3 h = cross(ray.dir, e1);
    const f32 a = dot(e0, h);

    if (fabs(a) < TRIANGLE_INTERSECT_EPSILON) {
        return TriangleIntersection::no_hit();
    }

    const f32 f = 1.0f / a;
    const Vector3 s = ray.pos - v0;
    const f32 u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return TriangleIntersection::no_hit();
    }

    const Vector3 q = cross(s, e0);
    const f32 v = f * dot(ray.dir, q);
    if (v < 0.0f || (u + v) > 1.0f) {
        return TriangleIntersection::no_hit();
    }

    const f32 t = f * dot(e1, q);

    if (t < TRIANGLE_INTERSECT_EPSILON) {
        return TriangleIntersection::no_hit();
    }

    return { t, u, v, 1 };
}

// Sphere intersect, obviously but I wanted to separate it from the triangle
// functions above :)
// ----------------------------------------------------------------------------

inline bool sphere_intersect(Ray ray, Vector3 center, f32 radius, f32* t_out) {
    const Vector3 o = ray.pos - center;
    const f32 b = dot(o, ray.dir);
    const f32 c = dot(o, o) - radius * radius;

    if (b > 0.0f && c > 0.0f) {
        return false;
    }

    const f32 d = b * b - c;

    if (d < 0.0f) {
        return false;
    }

    const f32 ds = sqrtf(d);

    const f32 t0 = -b - ds;
    const f32 t1 = -b + ds;

    f32 t_near, t_far;
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

    const f32 t_closest = (t_near < 0.0f) ? t_far : t_near;

    // NOTE
    // for now just report a true hit if at least one of the t is positive
    // this could also mean we are inside the sphere however.
    *t_out = t_closest;

    return true;
}

#endif // GUARD_STINGMATH_H
