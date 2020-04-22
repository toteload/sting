#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <math.h>
#include <initializer_list>






#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif






typedef   int8_t  i8;
typedef  uint8_t  u8;
typedef  int16_t i16;
typedef uint16_t u16;
typedef  int32_t i32;
typedef uint32_t u32;
typedef  int64_t i64;
typedef uint64_t u64;
typedef    float f32;
typedef   double f64;






constexpr f32 PI     = 3.14159265358979323846264338f;
constexpr f32 TWO_PI = 6.28318530717958647692528676f;





#define cast(Type, Expr) ((Type)(Expr))





union Vector2 {
    struct { f32 x, y; };
    f32 data[2];

              f32     length() const;
              void    set_length(f32 l);
              void    normalize();
              Vector2 normalized() const;
    constexpr Vector2 perpendicular_vector() const;
              bool    isnan() const;
};

constexpr inline Vector2 vec2(f32 x, f32 y);
          inline Vector2 rotate(const Vector2& a, f32 theta);
constexpr inline f32     dot(const Vector2& a, const Vector2& b);
constexpr inline Vector2 operator-(const Vector2& a, const Vector2& b);
constexpr inline Vector2 operator+(const Vector2& a, const Vector2& b);
constexpr inline Vector2 operator*(f32 s, const Vector2& a);
constexpr inline Vector2 operator/(const Vector2& a, f32 s);
          inline void    operator-=(Vector2& a, const Vector2& b);
          inline void    operator+=(Vector2& a, const Vector2& b);
          inline void    operator*=(Vector2& a, f32 s);
 
union Vector3 {
    struct { f32 x, y, z; };
    struct { f32 r, g, b; };
    f32 data[3];

    __device__                  f32     length() const;
    __device__                  void    normalize();
    __device__                  Vector3 normalized() const;
    __device__        constexpr Vector3 inverse() const;
    __device__                  f32&    operator[](size_t index);

    __device__ static Vector3 min(const Vector3& a, const Vector3& b);
    __device__ static Vector3 max(const Vector3& a, const Vector3& b);
};

__device__ constexpr inline Vector3 vec3(f32 a);
__device__ constexpr inline Vector3 vec3(f32 x, f32 y, f32 z);
__device__ constexpr inline f32     dot(const Vector3& a, const Vector3& b);
__device__ constexpr inline Vector3 cross(const Vector3& a, const Vector3& b);
__device__ constexpr inline Vector3 operator-(const Vector3& a, const Vector3& b);
__device__ constexpr inline Vector3 operator+(const Vector3& a, const Vector3& b);
__device__ constexpr inline Vector3 operator*(const Vector3& a, f32 s);
__device__ constexpr inline Vector3 operator*(f32 s, const Vector3& a);
__device__ constexpr inline Vector3 operator*(const Vector3& a, const Vector3& b);
__device__ constexpr inline Vector3 operator/(const Vector3& a, f32 s);
__device__           inline void    operator+=(Vector3& a, const Vector3& b);
__device__           inline void    operator*=(Vector3& a, const Vector3& b);
__device__           inline void    operator/=(Vector3& a, f32 s);
 
union alignas(16) Vector4 {
    struct { f32 x, y, z, w; };
    struct { f32 r, g, b, a; };
    f32 data[4];
};

__device__ constexpr inline Vector4 vec4(f32 x, f32 y, f32 z, f32 w);
__device__ constexpr inline Vector4 vec4(const Vector3& a, f32 w);
__device__ constexpr inline Vector4 operator+(const Vector4& a, const Vector4& b);
__device__ constexpr inline Vector4 operator*(f32 s, const Vector4& a);
__device__ constexpr inline Vector4 operator/(const Vector4& a, f32 s);
__device__           inline void    operator+=(Vector4& a, const Vector4& b);

struct RngXor32 {
    u32 state;

    __device__ RngXor32();
    __device__ RngXor32(u32 seed);

    __device__ u32 random_u32();

    // returns a f32 in the range [0.0f, 1.0f] inclusive
    __device__ f32 random_f32();
 
    __device__ u32 random_u32_max(u32 max);
};

__device__ inline RngXor32::RngXor32() : state(0) { }
__device__ inline RngXor32::RngXor32(u32 seed) : state(seed) { }

__device__ inline u32 RngXor32::random_u32() { 
    state ^= state << 13; 
    state ^= state >> 17; 
    state ^= state <<  5; 
    return state; 
}

__device__ inline f32 RngXor32::random_f32() {
    return random_u32() * 2.3283064365387e-10f;
}

__device__ inline u32 RngXor32::random_u32_max(u32 max) {
    return cast(u32, random_f32() * max);
}

constexpr u64 kilobytes(u64 x) { return 1024ull * x; }
constexpr u64 megabytes(u64 x) { return 1024ull * kilobytes(x); }
constexpr u64 gigabytes(u64 x) { return 1024ull * megabytes(x); }
    















inline f32     ease(f32 x);
inline f32     lerp(f32 start, f32 stop, f32 t);
inline Vector4 lerp(const Vector4& start, const Vector4& stop, f32 t);
 
template<typename T>
constexpr T max(const T& a, const T& b) {
    return (a > b) ? a : b;
}

template<typename T>
constexpr T min(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template<typename T>
constexpr T clamp(T t, T low, T high) {
    return min(high, max(low, t));
} 

template<typename T>
constexpr T min_recurse(const T* begin, const T* end) {
    // I would rather have this be an if/else but a constexpr function may only
    // have one return statement
    return (begin + 1 == end) ? (*begin) : min(*begin, min_recurse(begin + 1, end));
}

template<typename T>
constexpr T min(std::initializer_list<T> l) {
    return min_recurse(l.begin(), l.end());
}

template<typename T>
constexpr T max_recurse(const T* begin, const T* end) {
    return (begin + 1 == end) ? (*begin) : max(*begin, max_recurse(begin + 1, end));
}

template<typename T>
constexpr T max(std::initializer_list<T> l) {
    return max_recurse(l.begin(), l.end());
}



inline f32 Vector2::length() const {
    return sqrtf(x*x + y*y);
}

inline void Vector2::set_length(f32 s) {
    (*this) = (s / length()) * (*this);
}

inline void Vector2::normalize() {
    const f32 l = length();
    x /= l;
    y /= l;
}

inline Vector2 Vector2::normalized() const {
    Vector2 ret = *this;
    ret.normalize();
    return ret;
}

constexpr inline Vector2 Vector2::perpendicular_vector() const {
    return vec2(-y, x);
}

inline bool Vector2::isnan() const {
    return ::isnan(x) || ::isnan(y);
}

constexpr inline Vector2 vec2(f32 x, f32 y) {
    return {{ x, y, }};
}

inline Vector2 rotate(const Vector2& a, f32 theta) {
    return vec2(cosf(theta) * a.x - sinf(theta) * a.y,
                sinf(theta) * a.x + cosf(theta) * a.y);
}

constexpr inline f32 dot(const Vector2& a, const Vector2& b) {
    return a.x*b.x + a.y*b.y;
}

constexpr inline Vector2 operator-(const Vector2& a, const Vector2& b) {
    return vec2(a.x - b.x, a.y - b.y);
}

constexpr inline Vector2 operator+(const Vector2& a, const Vector2& b) {
    return vec2(a.x + b.x, a.y + b.y);
}
 
constexpr inline Vector2 operator*(f32 s, const Vector2& a) {
    return vec2(s * a.x, s * a.y);
}

constexpr inline Vector2 operator/(const Vector2& a, f32 s) {
    return vec2(a.x / s, a.y / s);
}

inline void operator+=(Vector2& a, const Vector2& b) {
    a = a + b;
}

inline void operator-=(Vector2& a, const Vector2& b) {
    a = a - b;
}

inline void operator*=(Vector2& a, f32 s) {
    a = s * a;
}

__device__ inline f32 Vector3::length() const {
    return sqrtf(dot(*this, *this));
}

__device__ inline void Vector3::normalize() {
    (*this) /= length();
}

__device__ inline Vector3 Vector3::normalized() const {
    return (*this) / length();
}

__device__ constexpr inline Vector3 Vector3::inverse() const {
    return vec3(1.0f / x, 1.0f / y, 1.0f / z);
}
__device__ f32& Vector3::operator[](size_t index) {
    return data[index];
}

__device__ constexpr inline Vector3 vec3(f32 a) {
    return vec3(a, a, a);
}

__device__ constexpr inline Vector3 vec3(f32 x, f32 y, f32 z) {
    return {{ x, y, z, }};
}

__device__ constexpr inline f32 dot(const Vector3& a, const Vector3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ constexpr inline Vector3 cross(const Vector3& a, const Vector3& b) {
    return vec3(a.y*b.z - a.z*b.y,
                a.z*b.x - a.x*b.z,
                a.x*b.y - a.y*b.x);
}

__device__ constexpr inline Vector3 operator-(const Vector3& a, const Vector3& b) {
    return vec3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ constexpr inline Vector3 operator+(const Vector3& a, const Vector3& b) {
    return vec3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ constexpr inline Vector3 operator*(const Vector3& a, f32 s) {
    return vec3(s*a.x, s*a.y, s*a.z);
}

__device__ constexpr inline Vector3 operator*(f32 s, const Vector3& a) {
    return vec3(s*a.x, s*a.y, s*a.z);
}

__device__ constexpr inline Vector3 operator*(const Vector3& a, const Vector3& b) {
    return vec3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ constexpr inline Vector3 operator/(const Vector3& a, f32 s) {
    return vec3(a.x / s, a.y / s, a.z / s);
}

__device__ inline void operator+=(Vector3& a, const Vector3& b) {
    a = a + b;
}

__device__ inline void operator*=(Vector3& a, const Vector3& b) {
    a = a * b;
}

__device__ inline void operator/=(Vector3& a, f32 s) {
    a = a / s;
}

__device__ Vector3 Vector3::min(const Vector3& a, const Vector3& b) {
     return vec3(::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z));
}

__device__ Vector3 Vector3::max(const Vector3& a, const Vector3& b) {
     return vec3(::max(a.x, b.x), ::max(a.y, b.y), ::max(a.z, b.z));
}

__device__ constexpr inline Vector4 vec4(f32 x, f32 y, f32 z, f32 w) {
    return {{ x, y, z, w, }};
}

__device__ constexpr inline Vector4 vec4(const Vector3& a, f32 w) {
    return vec4(a.x, a.y, a.z, w);
}

__device__ constexpr inline Vector4 operator+(const Vector4& a, const Vector4& b) {
    return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ constexpr inline Vector4 operator*(f32 s, const Vector4& a) {
    return vec4(s * a.x, s * a.y, s * a.z, s * a.w);
} 

__device__ constexpr inline Vector4 operator/(const Vector4& a, f32 s) {
    return vec4(a.x / s, a.y / s, a.z / s, a.w / s);
}

__device__ inline void operator+=(Vector4& a, const Vector4& b) {
    a = a + b;
}

inline f32 ease(f32 x) {
    return (x * x) * (3 - 2 * x);
}

inline f32 lerp(f32 start, f32 stop, f32 t) {
    return start + (stop - start) * t;
}

inline Vector4 lerp(const Vector4& start, const Vector4& stop, f32 t) {
    return vec4(lerp(start.x, stop.x, t),
                lerp(start.y, stop.y, t),
                lerp(start.z, stop.z, t),
                lerp(start.w, stop.w, t));
}

struct Arena {
    u8* base;
    u8* at;
    u8* end;

    void init(void* mem, u32 size);
    void reset();
    u32 capacity();
    u32 size();
    u32 bytes_left();

    char* push_string(const char* fmt, ...);

    template<typename T>
    T* alloc(u32 num_of_t);
};

void Arena::init(void* mem, u32 size) {
    base = cast(u8*, mem);
    at   = cast(u8*, mem);
    end  = cast(u8*, mem) + size;
}

void Arena::reset() {
    at = base;
}

u32 Arena::capacity() {
    return cast(u32, end - base);
}

u32 Arena::size() {
    return cast(u32, at - base);
}

u32 Arena::bytes_left() {
    return cast(u32, end - at);
}

char* Arena::push_string(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    u32 written = vsnprintf(cast(char*, at), bytes_left(), fmt, args);
    char* str = cast(char*, at);
    at += written + 1;
    return str;
}

template<typename T>
T* alloc(u32 num_of_t) {
    u32 bytes_needed = num_of_t * sizeof(T);
    if (bytes_needed > bytes_left()) {
        return NULL;
    }

    T* ret = cast(T*, at);
    at += bytes_needed;

    return ret;
}
     
inline void* read_file(const char* filename, u64* size = NULL) {
    FILE* f = fopen(filename, "rb");
    if (!f) { return NULL; }
    fseek(f, 0, SEEK_END);
    u64 file_size = ftell(f);
    void* data = malloc(file_size + 1);
    if (!data) { return NULL; }
    fseek(f, 0, SEEK_SET);
    fread(data, 1, file_size, f);
    cast(u8*, data)[file_size] = '\0';
    fclose(f);
    if (size) { *size = file_size; }
    return data;
}

// When this function is finished all the items for which the predicate was
// not met are at the start of the array and all the items for which the
// predicate was met are at the end of the array.
// Returns the index of the first item for which the predicate was met.
template<typename T, typename Predicate>
u32 partition_array(T* items, u32 size, Predicate pred) {
    u32 last_good = size;
    for (u32 i = 0; i < size && i < last_good; i++) {
        if (pred(items[i])) {
            while (last_good-- > 0 && last_good > i && pred(items[last_good])) { }

            if (last_good <= i) {
                break;
            }

            std::swap(items[i], items[last_good]);
        }
    }

    return last_good;
}
  
inline bool is_whitespace(char c) {
    return (c == ' ' || c == '\t' || c == '\n' || c =='\r');
}

inline bool is_digit(char c) {
    return (c >= '0' && c <= '9');
}

inline bool is_lower(char c) {
    return (c >= 'a' && c <= 'z');
}

inline bool is_upper(char c) {
    return (c >= 'A' && c <= 'Z');
}

inline bool is_alpha(char c) {
    return (is_lower(c) || is_upper(c));
}

inline bool is_hex(char c) {
    return (is_digit(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
}

inline bool is_octal(char c) {
    return (c >= '0' && c <= '7');
}
                                 
inline i64 str_to_i64(const char* str, u32 len) {
    i64 base = 10;
    i64 sign = 1;

    if (str[0] == '+') {            str++; len--; }
    if (str[0] == '-') { sign = -1; str++; len--; }

    if (len == 0) { return 0; }

    if (str[0] == '0') {
        if (len == 1) { return 0; }

        switch (str[1]) {
            case 'x': case 'X': base = 16; break;
            case 'b': case 'B': base =  2; break;
            default:            base =  8; break;
        }

        str += 2;
        len -= 2;
    }

    i64 ret = 0;
    for (u32 i = 0; i < len; i++) {
        const char c = str[len-i-1];
        if (c == '_') { continue; }
        i64 v;
        if (is_digit(c)) {
            v = c - '0';
        } else {
            v = is_upper(c) ? (c - 'A') : (c - 'a');
        }
        ret = ret * base + v;
    }

    return sign * ret;
}
 
template<typename T, u32 Capacity>
struct Stack {
    T data[Capacity];
    u32 top;

    __device__ Stack() : top(0) { }
    __device__ Stack(std::initializer_list<T> l) : top(l.size()) {
        for (u32 i = 0; i < l.size(); i++) {
            data[i] = l.begin()[i];
        }
    }

    __device__ bool is_empty() const { return top == 0; }
    __device__ u32 capacity() const { return Capacity; }
    __device__ u32 size() const { return top; }
    __device__ void push(T v) { data[top++] = v; }
    __device__ T pop() { return data[--top]; }
    __device__ T peek() const { return data[top-1]; }
};

// after calling this a will be min(a, b) and b max(a, b)
template<typename T>
void sort_compare_and_swap(T& a, T& b) {
    if (a > b) { std::swap(a, b); }
}

template<typename T>
void sorting_network(T* items, u32 size) { 
    switch (size) {
    case 0: case 1: break;
    case 2: {
    sort_compare_and_swap(items[0], items[1]);
    } break;
    case 3: {
    sort_compare_and_swap(items[0], items[1]);
    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[0], items[1]);
    } break;
    case 4: {
    sort_compare_and_swap(items[0], items[1]);
    sort_compare_and_swap(items[2], items[3]);

    sort_compare_and_swap(items[0], items[2]);
    sort_compare_and_swap(items[1], items[3]);

    sort_compare_and_swap(items[2], items[3]);
    } break;
    case 5: {
    sort_compare_and_swap(items[0], items[3]);
    sort_compare_and_swap(items[1], items[4]);

    sort_compare_and_swap(items[0], items[2]);
    sort_compare_and_swap(items[1], items[3]);
    
    sort_compare_and_swap(items[0], items[1]);
    sort_compare_and_swap(items[2], items[4]);

    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[3], items[4]);

    sort_compare_and_swap(items[2], items[3]);
    } break;
    case 6: {
    sort_compare_and_swap(items[0], items[5]);
    sort_compare_and_swap(items[1], items[3]);
    sort_compare_and_swap(items[2], items[4]);

    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[3], items[4]);

    sort_compare_and_swap(items[0], items[3]);
    sort_compare_and_swap(items[2], items[5]);

    sort_compare_and_swap(items[0], items[1]);
    sort_compare_and_swap(items[2], items[3]);
    sort_compare_and_swap(items[4], items[5]);

    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[3], items[4]);
    } break;
    case 7: {
    sort_compare_and_swap(items[0], items[6]);
    sort_compare_and_swap(items[2], items[3]);
    sort_compare_and_swap(items[4], items[5]);

    sort_compare_and_swap(items[0], items[2]);
    sort_compare_and_swap(items[1], items[4]);
    sort_compare_and_swap(items[3], items[6]);

    sort_compare_and_swap(items[0], items[1]);
    sort_compare_and_swap(items[2], items[5]);
    sort_compare_and_swap(items[3], items[4]);

    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[4], items[6]);

    sort_compare_and_swap(items[2], items[3]);
    sort_compare_and_swap(items[4], items[5]);

    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[3], items[4]);
    sort_compare_and_swap(items[5], items[6]);
    } break;
    case 8: {
    sort_compare_and_swap(items[0], items[2]);
    sort_compare_and_swap(items[1], items[3]);
    sort_compare_and_swap(items[4], items[6]);
    sort_compare_and_swap(items[5], items[7]);

    sort_compare_and_swap(items[0], items[4]);
    sort_compare_and_swap(items[1], items[5]);
    sort_compare_and_swap(items[2], items[6]);
    sort_compare_and_swap(items[3], items[7]);

    sort_compare_and_swap(items[0], items[1]);
    sort_compare_and_swap(items[2], items[3]);
    sort_compare_and_swap(items[4], items[5]);
    sort_compare_and_swap(items[6], items[7]);

    sort_compare_and_swap(items[2], items[4]);
    sort_compare_and_swap(items[3], items[5]);

    sort_compare_and_swap(items[1], items[4]);
    sort_compare_and_swap(items[3], items[6]);

    sort_compare_and_swap(items[1], items[2]);
    sort_compare_and_swap(items[3], items[4]);
    sort_compare_and_swap(items[5], items[6]);
    } break;
    }
    default: assert(0); break;
}
