struct Camera {
    vec3 pos;
    vec3 dir;

    float inclination; // range [0, PI)
    float azimuth; // range [0, 2*PI)

    vec3 u, v, w;

    static Camera create(vec3 pos, vec3 up, vec3 at, float width, float height);
};
