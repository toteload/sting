struct Camera {
    vec3 pos;
    vec3 dir;

    float inclination; // range [0, PI)
    float azimuth; // range [0, 2*PI)

    vec3 u, v, w;
};
