#ifndef GUARD_STINGMATH_H
#define GUARD_STINGMATH_H

#define _USE_MATH_DEFINES // for PI constants
#include <math.h>
#include <float.h>
#include <initializer_list>

union vec4;

union vec3 {
    struct { f32 x, y, z; };
    struct { f32 r, g, b; };
    f32 fields[3];

    __host__ __device__ vec3() { }
    __host__ __device__ vec3(f32 a): x(a), y(a), z(a) { }
    __host__ __device__ vec3(f32 x, float y, float z): x(x), y(y), z(z) { }
    __host__ __device__ vec3(const vec4& v);

    __host__ __device__ inline f32  length_squared() const;
    __host__ __device__ inline f32  length() const;
    __host__ __device__ inline vec3 normalize() const;
    __host__ __device__ inline vec3 inverse() const;
    __host__ __device__ inline f32  operator[](size_t i) const;

    __host__ __device__ inline static vec3 zero();
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
    struct { f32 x, y, z, w; };
    struct { f32 r, g, b, a; };
    f32 fields[4];

    __host__ __device__ vec4() { }
    __host__ __device__ vec4(f32 v) : x(v), y(v), z(v), w(v) { }
    __host__ __device__ vec4(f32 x, float y, float z, float w) : x(x), y(y), z(z), w(w) { }
    __host__ __device__ vec4(const vec3& v, f32 w) : x(v.x), y(v.y), z(v.z), w(w) { }
};

__device__ inline vec4 operator*(f32 s, vec4 v);
__device__ inline vec4 operator+(const vec4&a, const vec4& b);
__device__ inline void operator+=(vec4& a, const vec4& b);
__device__ inline vec4 operator/(const vec4& a, f32 s);

struct alignas(16) Ray {
    vec3 pos; f32 tmin;
    vec3 dir; f32 tmax;

    __device__ Ray(vec3 pos, vec3 dir) : pos(pos), tmin(FLT_MIN), dir(dir), tmax(FLT_MAX) { }
    __device__ Ray(vec3 pos, vec3 dir, f32 tmin, f32 tmax) : pos(pos), tmin(tmin), dir(dir), tmax(tmax) { }
};

// Random number generation
// ----------------------------------------------------------------------------

struct RngXor32 {
    u32 state;

    __device__ RngXor32(u32 seed) : state(seed) { }

    __device__ u32 random_u32() { 
        state ^= state << 13; 
        state ^= state >> 17; 
        state ^= state <<  5; 
        return state; 
    }

    // returns a f32 in the range [0.0f, 1.0f] inclusive
    __device__ f32 random_f32() {
        return random_u32() * 2.3283064365387e-10f;
    }
 
    __device__ u32 random_u32_max(u32 max) {
        return cast(u32, random_f32() * max);
    }
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
inline T clamp(const T& x, const T& a, const T& b) {
    return min(b, max(x, a));
}

inline f32 degrees_to_radians(f32 d) {
    return (d / 180.0f) * M_PI;
}

// inclination and azimuth in radians
inline __host__ __device__ 
void cartesian_to_spherical(const vec3& v, f32* inclination, f32* azimuth) {
    const f32 r = v.length();
    *inclination = acosf(v.y / r); // in range [0, pi]
    *azimuth = atan2f(v.z, v.x); // in range [-pi, pi]
}

// inclination and azimuth in radians
inline __host__ __device__ 
vec3 spherical_to_cartesian(f32 inclination, f32 azimuth) {
    return { sinf(inclination) * cosf(azimuth),
             cosf(inclination),
             sinf(inclination) * sinf(azimuth) };
}

__device__ inline
void build_orthonormal_basis(const vec3& n, vec3* t, vec3* b) {
    *t = (fabs(n.x) > fabs(n.y)) ? vec3(n.z, 0.0f, -n.x).normalize() : 
                                   vec3(0.0f, -n.z, n.y).normalize();
    *b = cross(n, *t);
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

__host__ __device__ inline vec3::vec3(const vec4& v) : x(v.x), y(v.y), z(v.z) { }
__host__ __device__ inline f32  vec3::length_squared() const { return x*x + y*y + z*z; }
__host__ __device__ inline f32  vec3::length() const { return sqrtf(x*x + y*y + z*z); }
__host__ __device__ inline vec3 vec3::normalize() const { const f32 l = length(); return {x / l, y / l, z / l }; }
__host__ __device__ inline vec3 vec3::inverse() const { return { 1.0f / x, 1.0f / y, 1.0f / z }; }
__host__ __device__ inline f32  vec3::operator[](size_t i) const { return fields[i]; }

__host__ __device__ inline vec3 vec3::zero() { return vec3(0.0f, 0.0f, 0.0f); }

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
f32 dot(vec3 a, vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline 
vec3 cross(vec3 a, vec3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x, };
}

__host__ __device__ inline
vec3 operator*(f32 s, vec3 v) {
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

// This doesn't work (yet)
#if 0
inline u32 pack_normal(vec3 n) {
    const f32 f = 65535.0f / max(sqrtf(8.0f * n.z + 8.0f), 0.0001f);
    return cast(u32, n.x * f + 32767.0f) + (cast(u32, n.y * f + 32767.0f) << 16);
}

inline __device__ vec3 unpack_normal(u32 p) {
    vec4 nn(cast(f32, p & 65335) * (2.0f / 65335.0f) - 1.0f,
            cast(f32, p >> 16) * (2.0f / 65335.0f) - 1.0f,
            -1.0f,
            -1.0f);

    const f32 l = dot(vec3(nn.x, nn.y, nn.z), vec3(-nn.x, -nn.y, -nn.w));
    nn.z = l;

    const f32 ll = sqrtf(l);
    nn.x *= ll;
    nn.y *= ll;

    return vec3(nn.x, nn.y, nn.z) * 2.0f + vec3(0.0f, 0.0f, -1.0f);
}
#endif

// triangle functions
// ----------------------------------------------------------------------------

__host__ __device__ inline
f32 triangle_solid_angle(const vec3& a, const vec3& b, const vec3& c) {
    const f32 da = a.length();
    const f32 db = b.length();
    const f32 dc = c.length();

    return 2.0f * atan2f(dot(a, cross(b, c)), 
                         //da * (db * (dc * (1 + dot(a, b)) + dot(a, c)) + dot(b, c)));
                         da*db*dc + dot(a, b)*dc + dot(a, c)*db + dot(b, c)*da);
}

__host__ __device__ inline
f32 triangle_solid_angle_hemisphere(const vec3& a, const vec3& b, const vec3& c) {
    const u32 a_below = a.y < 0;
    const u32 b_below = b.y < 0;
    const u32 c_below = c.y < 0;

    const u32 below_hemisphere = a_below + b_below + c_below;

    switch (below_hemisphere) {
    case 3: { return 0.0f; }
    case 2: { 
        // two points of the triangle are below the hemisphere so the part of
        // the triangle that is in the hemisphere is shaped like a triangle
        vec3 top, p0, p1;
        if (!a_below) { top = a; p0 = b; p1 = c; }
        if (!b_below) { top = b; p0 = c; p1 = a; }
        if (!c_below) { top = c; p0 = a; p1 = b; }

        const vec3 e0 = (p0 - top);
        const vec3 e1 = (p1 - top);

        const vec3 bb = (-top.y / e0.y) * e0;
        const vec3 cc = (-top.y / e1.y) * e1;

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

        vec3 bot, p0, p1;
        if (a_below) { bot = a; p0 = b; p1 = c; }
        if (b_below) { bot = b; p0 = c; p1 = a; }
        if (c_below) { bot = c; p0 = a; p1 = b; }

        const vec3 e0 = (p0 - bot);
        const vec3 e1 = (p1 - bot);

        const vec3 bb = (-bot.y / e0.y) * e0;
        const vec3 cc = (-bot.y / e1.y) * e1;

        return fabs(triangle_solid_angle(a, b, c)) - fabs(triangle_solid_angle(bot, bb, cc));
    } break;
    case 0: { return triangle_solid_angle(a, b, c); }

    // Just to shut up compiler
    default: { return 0; }
    } 
}

__host__ __device__ inline
vec3 triangle_normal(vec3 v0, vec3 v1, vec3 v2) {
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    return cross(e0, e1).normalize();
}

__device__ inline
vec3 triangle_normal_lerp(vec3 n0, vec3 n1, vec3 n2, f32 u, float v) {
    const f32 w = 1.0f - u - v;
    return w * n0 + u * n1 + v * n2;
}

__host__ inline
f32 triangle_area(const vec3& p0, const vec3& p1, const vec3& p2) {
    const f32 a = (p0 - p1).length();
    const f32 b = (p1 - p2).length();
    const f32 c = (p2 - p0).length();
    const f32 s = (a + b + c) / 2.0f;
    return sqrtf(s * (s - a) * (s - b) * (s - c));
}

__device__ inline
vec3 triangle_random_point(const vec3& p0, const vec3& p1, const vec3& p2, f32 r1, f32 r2) {
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
TriangleIntersection triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2) {
    // This value has been empirically found, there is a good chance another value is better :)
    const f32 TRIANGLE_INTERSECT_EPSILON = 1e-6f;

    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v0;

    const vec3 h = cross(ray.dir, e1);
    const f32 a = dot(e0, h);

    if (fabs(a) < TRIANGLE_INTERSECT_EPSILON) {
        return TriangleIntersection::no_hit();
    }

    const f32 f = 1.0f / a;
    const vec3 s = ray.pos - v0;
    const f32 u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return TriangleIntersection::no_hit();
    }

    const vec3 q = cross(s, e0);
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

inline bool sphere_intersect(Ray ray, vec3 center, f32 radius, float* t_out) {
    const vec3 o = ray.pos - center;
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
