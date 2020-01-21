#ifndef GUARD_VECMATH_H
#define GUARD_VECMATH_H

union vec3 {
    struct { float x, y, z; };
    struct { float r, g, b; };
    float fields[3];
};

union vec4 {
    struct { float x, y, z, w; };
    struct { float r, g, b, a; };
    float fields[4];
};

struct Ray {
    vec3 pos;
    vec3 dir;   
};

#endif // GUARD_VECMATH_H
