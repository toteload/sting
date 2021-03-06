#ifndef GUARD_CAMERA_H
#define GUARD_CAMERA_H

#include "stingmath.h"

struct HitRecord {
    Vector3 pos;
    float t;
    Vector3 normal;
};

struct PointCamera {
    Vector3 pos;

    float inclination; // range [0, PI)
    float azimuth; // range [0, 2*PI)

    float width, height; 
    float plane_distance;

    Vector3 u, v, w;

    PointCamera(Vector3 pos, Vector3 up, Vector3 at, float width, float height, float plane_distance) {
        const Vector3 forward = (at - pos).normalized();

        cartesian_to_spherical(forward, &inclination, &azimuth);

        w = forward;
        u = cross(w, up).normalized();
        v = cross(w, u).normalized();

        this->pos = pos;
        this->width = width;
        this->height = height;
        this->plane_distance = plane_distance;
    }

    void update_uvw() {
        // inclination cannot be exactly 0 or pi, because then our w vector
        // will be perpendicular to our up vector which causes some camera 
        // glitching. This is easy fix :)
        constexpr f32 EPSILON = 0.000001f;
        inclination = clamp<float>(inclination, EPSILON, M_PI - EPSILON);

        if (azimuth > 2.0f * M_PI) { azimuth -= 2.0f * M_PI; }
        if (azimuth < 0.0f)        { azimuth += 2.0f * M_PI; }

        w = spherical_to_cartesian(inclination, azimuth);
        u = cross(w, vec3(0.0f, 1.0f, 0.0f)).normalized();
        v = cross(w, u).normalized();
    }

    // uu and vv are in range (-1.0f, 1.0f)
    __device__ Ray create_ray(float uu, float vv) { 
        const float px = 0.5f * uu * width;
        const float py = 0.5f * vv * height;

        const Vector3 p = plane_distance * w + px * u + py * v;

        return Ray(pos, p.normalized());
    }
};

#endif // GUARD_CAMERA_H
