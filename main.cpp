#include "ext/SDL2-2.0.10/include/SDL.h"
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <Windows.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdint.h>
#include <stdio.h>

#include "common.h"
#include "glext.h"
#include "load_gl_extensions.h"
#include "stingmath.h"
#include "camera.h"
#include "bvh.h"
#include "bvh.cpp"
#include "fast_obj.h"
#include "stb_image.h"
#include "meshloader.h"

const uint32_t STING_VERSION_MAJOR = 0;
const uint32_t STING_VERSION_MINOR = 1;
const uint32_t STING_VERSION_REVISION = 1;

inline cudaError cuda_err_check(cudaError err, const char* file, int line, bool abort) {
    (void)abort;
    if (err != cudaSuccess) {
        fprintf(stderr, "[cuda check failed] error code: %d, message: %s, file: %s, line: %d\n",
                err, cudaGetErrorString(err), file, line);
    }

    return err;
}

#define CUDA_CHECK(...) cuda_err_check(__VA_ARGS__, __FILE__, __LINE__, false)

const u32 SCREEN_WIDTH  = 960;
const u32 SCREEN_HEIGHT = 540;

struct Keymap {
    u32 w : 1;
    u32 s : 1;
    u32 a : 1;
    u32 d : 1;
    u32 up: 1;
    u32 down: 1;
    u32 left: 1;
    u32 right: 1;

    static Keymap empty() {
        Keymap map;
        memset(&map, 0, sizeof(map));
        return map;
    }
};

void render(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, vec4 const * skybox,
            vec4* buffer, u32 width, u32 height, u32 framenum);
void render_normal(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
                   vec4* buffer, u32 width, u32 height, u32 framenum);
void render_buffer_to_screen(cudaArray_const_t array, vec4* screen_buffer, uint32_t width, uint32_t height);
void accumulate(vec4* frame_buffer, vec4* accumulator, vec4* screen_buffer, u32 width, u32 height, u32 acc_frame);

void render_nee(BVHNode const * bvh, RenderTriangle const * triangles, 
                u32 const * lights, u32 light_count,
                PointCamera camera, vec4 const * skybox,
                vec4* buffer, u32 width, u32 height, u32 framenum);

#if 1
std::vector<RenderTriangle> generate_sphere_mesh(u32 rows, u32 columns, f32 radius) {
    if (rows < 2 || columns < 3) {
        return { };
    }

    std::vector<vec3> pts;
    std::vector<vec3> normals;

    for (u32 i = 1; i < rows; i++) {
        for (u32 j = 0; j < columns; j++) {
            const f32 phi = i * M_PI / cast(f32, rows);
            const f32 theta = (cast(f32, j) / cast(f32, columns)) * 2.0f * M_PI;
            const vec3 n = spherical_to_cartesian(phi, theta);
            pts.push_back(radius * n);
            normals.push_back(n);
        }
    }

    const size_t top = pts.size();
    pts.push_back(radius * vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));

    const size_t bottom = pts.size();
    pts.push_back(radius * vec3(0.0f, -1.0f, 0.0f));
    normals.push_back(vec3(0.0f, -1.0f, 0.0f));

    std::vector<RenderTriangle> triangles;

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;
        triangles.push_back(RenderTriangle(    pts[top],     pts[i],     pts[inext],
                                           normals[top], normals[i], normals[inext]));
    }

    for (u32 r = 0; r < rows - 2; r++) {
        for (u32 c = 0; c < columns; c++) {
            const u32 cnext = (c + 1) % columns;

            const u32 rowi = r * columns;
            const u32 rowinext = (r + 1) * columns;

#if 1
            triangles.push_back(RenderTriangle(    pts[rowi + c],     pts[rowinext + c],     pts[rowi + cnext],
                                               normals[rowi + c], normals[rowinext + c], normals[rowi + cnext]));
#endif

            //triangles.push_back(RenderTriangle(    pts[rowi + c],     pts[rowinext + c],     pts[rowi + cnext]));
            //triangles.push_back(RenderTriangle(    pts[rowinext + c],     pts[rowinext + cnext],     pts[rowi + cnext]));
#if 1
            triangles.push_back(RenderTriangle(    pts[rowinext + c],     pts[rowinext + cnext],     pts[rowi + cnext],
                                               normals[rowinext + c], normals[rowinext + cnext], normals[rowi + cnext]));
#endif
        }
    }

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;
        const u32 rowi = (rows - 2) * columns;

        triangles.push_back(RenderTriangle(    pts[bottom],     pts[rowi + inext],     pts[rowi + i],
                                           normals[bottom], normals[rowi + inext], normals[rowi + i]));
    }

    for (u32 i = 0; i < triangles.size(); i++) {
        triangles[i].material = MATERIAL_DIFFUSE;
        triangles[i].colorr = 0.8f;
        triangles[i].colorg = 0.8f;
        triangles[i].colorb = 0.8f;
    }

    return triangles;
}
#endif

bool verify_bvh(BVHNode const * bvh, u32 bvh_size, u32 prim_count) {
    std::vector<bool> check;
    check.resize(prim_count, false);
    for (u32 i = 0; i < bvh_size; i++) {
        if (bvh[i].is_leaf()) {
            for (u32 j = bvh[i].left_first; j < bvh[i].left_first + bvh[i].count; j++) {
                check[j] = true;
            }
        }
    }

    bool found_all_prims = true;
    for (u32 i = 0; i < prim_count; i++) {
        if (!check[i]) {
            found_all_prims = false;
            printf("prim %d not found in bvh!!!\n", i);
        }
    }

    return found_all_prims;
}

int main(int argc, char** args) {
    UNUSED(argc); UNUSED(args);

    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);

    // NOTE: when these are set it cannot find a suitable opengl context
    //SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 32);
    //SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 32);
    //SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 32);
    //SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 32);

    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    char name_buffer[128];
    snprintf(name_buffer, 128, "Sting %d.%d.%d", STING_VERSION_MAJOR, STING_VERSION_MINOR, STING_VERSION_REVISION);

    SDL_Window* window = SDL_CreateWindow(name_buffer,
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          SCREEN_WIDTH,
                                          SCREEN_HEIGHT,
                                          SDL_WINDOW_OPENGL);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);

    if (!load_gl_extensions()) {
        return 1;
    }

    // enable/disable vsync
    SDL_GL_SetSwapInterval(0);

    // CUDA/OpenGL interop setup
    // --------------------------------------------------------------------- //
    GLuint gl_frame_buffer, gl_render_buffer;
    glCreateRenderbuffers(1, &gl_render_buffer);
    glCreateFramebuffers(1, &gl_frame_buffer);
    glNamedFramebufferRenderbuffer(gl_frame_buffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gl_render_buffer);
    glNamedRenderbufferStorage(gl_render_buffer, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT);

    cudaGraphicsResource* graphics_resource;
    cudaGraphicsGLRegisterImage(&graphics_resource,
                                gl_render_buffer, GL_RENDERBUFFER, 
                                cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &graphics_resource);

    cudaArray* screen_array;
    cudaGraphicsSubResourceGetMappedArray(&screen_array, graphics_resource, 0, 0);
    cudaGraphicsUnmapResources(1, &graphics_resource);

    // Allocate the buffers
    vec4* screen_buffer, *frame_buffer, *accumulator, *normal_buffer;
    cudaMalloc(&screen_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4));
    cudaMalloc(&frame_buffer,  SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4));
    cudaMalloc(&accumulator,   SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4));
    cudaMalloc(&normal_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4));

    // Loading in triangle mesh, building bvh and uploading to GPU
    // --------------------------------------------------------------------- //

    Scene scene;
    
    std::vector<RenderTriangle> triangles;

#if 1
    auto sphere_mesh = generate_sphere_mesh(36, 36, 0.2f);
    for (u32 i = 0; i < sphere_mesh.size(); i++) {
        sphere_mesh[i].material = MATERIAL_MIRROR;
    }
    triangles.insert(triangles.end(), sphere_mesh.begin(), sphere_mesh.end());
#endif

#if 1
    auto light0 = RenderTriangle(vec3(0.0f, 0.4f, 0.0f),
                                 vec3(0.3f, 0.4f, 0.0f),
                                 vec3(0.0f, 0.5f, 0.3f));
    light0.material = MATERIAL_EMISSIVE;
    light0.light_intensity = 10.0f;
    triangles.push_back(light0);

    auto light1 = RenderTriangle(vec3(0.0f, -0.4f, 0.0f),
                                 vec3(0.3f, -0.4f, 0.0f),
                                 vec3(0.0f, -0.5f, 0.3f));
    light1.material = MATERIAL_EMISSIVE;
    light1.light_intensity = 10.0f;
    triangles.push_back(light1);

    auto light2 = RenderTriangle(vec3(0.0f, 0.0f, -0.8f),
                                 vec3(0.3f, 0.0f, -0.8f),
                                 vec3(0.3f, 0.5f, -0.8f));
    light2.material = MATERIAL_EMISSIVE;
    light2.light_intensity = 10.0f;
    triangles.push_back(light2);
#endif

#if 0
    fastObjMesh* mesh = fast_obj_read("Thai_Buddha.obj");

    printf("Mesh has %d vertices, and %d normals\n", mesh->position_count, mesh->normal_count);

    if (!mesh) {
        printf("Failed to load mesh...\n");
        return 1;
    }

    // Loop over all the groups in the mesh
    for (uint32_t i = 0; i < mesh->group_count; i++) {
        fastObjGroup group = mesh->groups[i];

        uint32_t vertex_index = 0;

        // Loop over all the faces in this group
        for (uint32_t j = 0; j < group.face_count; j++) {
            const uint32_t vertex_count = mesh->face_vertices[group.face_offset + j];

            if (vertex_count != 3) {
                printf("Found a face that is not a triangle...\n");
                continue;
            }

            vec3 vertices[3];

            // Loop over all the vertices in this face
            for (uint32_t k = 0; k < vertex_count; k++) {
                fastObjIndex index = mesh->indices[group.index_offset + vertex_index];

                vertices[k] = vec3(mesh->positions[3 * index.p + 0], 
                                   mesh->positions[3 * index.p + 1], 
                                   mesh->positions[3 * index.p + 2]);

                vertex_index++;
            }
            auto tri = RenderTriangle(vertices[0], vertices[1], vertices[2]);
            tri.material = MATERIAL_MIRROR;
            triangles.push_back(tri);
        }
    }

    fast_obj_destroy(mesh);
#endif

#if 0
    {
        const vec3 n = triangles[0].face_normal;
        const u32 packed = pack_normal(n);
        const vec3 unpacked = unpack_normal(packed);
        printf("normal: %f, %f, %f, packed: %d, unpacked: %f, %f, %f\n", 
               n.x, n.y, n.z, 
               packed, 
               unpacked.x, unpacked.y, unpacked.z);
    }
#endif

    printf("%llu RenderTriangles created...\n", triangles.size());
    
    std::vector<BVHNode> bvh = build_bvh_for_triangles(triangles.data(), triangles.size());

    std::vector<u32> lights;
    for (u32 i = 0; i < triangles.size(); i++) {
        if (triangle.material == MATERIAL_EMISSIVE) {
            lights.push_back(i);
        }
    }

    printf("Created %llu bvh nodes...\n", bvh.size());
    verify_bvh(bvh.data(), bvh.size(), triangles.size());

    RenderTriangle* gpu_triangles;
    CUDA_CHECK(cudaMalloc(&gpu_triangles, triangles.size() * sizeof(RenderTriangle)));

    BVHNode* gpu_bvh;
    CUDA_CHECK(cudaMalloc(&gpu_bvh, bvh.size() * sizeof(BVHNode)));

    CUDA_CHECK(cudaMemcpy(gpu_triangles, triangles.data(), triangles.size() * sizeof(RenderTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_bvh, bvh.data(), bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    // Skybox loading
    // --------------------------------------------------------------------- //
#if 1
    i32 hdr_width, hdr_height, hdr_channels;
    u8* skybox_image = stbi_load("cloudysky_4k.jpg", &hdr_width, &hdr_height, &hdr_channels, 4);
    if (!skybox_image) {
        printf("Failed to load sky dome...\n");
    }

    printf("Loaded skybox texture, width: %d, height: %d\n", hdr_width, hdr_height);

    vec4* skybox_buffer;
    CUDA_CHECK(cudaMalloc(&skybox_buffer, hdr_width * hdr_height * sizeof(vec4)));

    vec4* skybox = cast(vec4*, malloc(hdr_width * hdr_height * sizeof(vec4)));
    for (u32 i = 0; i < hdr_width * hdr_height; i++) {
        skybox[i] = vec4(skybox_image[i * 4 + 0] / 255.0f,
                         skybox_image[i * 4 + 1] / 255.0f,
                         skybox_image[i * 4 + 2] / 255.0f,
                         1.0f);
    }

    CUDA_CHECK(cudaMemcpy(skybox_buffer, skybox, hdr_width * hdr_height * sizeof(vec4), cudaMemcpyHostToDevice));

    free(skybox);
    stbi_image_free(skybox_image);
#endif

    // Setting the camera and keymap
    // --------------------------------------------------------------------- //
    PointCamera camera(vec3(-0.5f, 0.5f, 0.5f), // position
                       vec3(0, 1, 0), // up
                       vec3(0, 0, 0), // at
                       SCREEN_WIDTH, SCREEN_HEIGHT, 
                       600);

    Keymap keymap = Keymap::empty();

    SDL_SetRelativeMouseMode(SDL_TRUE);

    uint64_t frame_count = 0;
    u32 acc_frame = 0;

    const uint64_t frequency = SDL_GetPerformanceFrequency();
    uint64_t previous_time = SDL_GetPerformanceCounter();

    bool running = true;
    while (running) {
        uint64_t current_time = SDL_GetPerformanceCounter();
        const uint64_t time_diff = current_time - previous_time;
        const float seconds = cast(float, time_diff) / frequency;
        previous_time = current_time;

        i32 mouse_x = 0, mouse_y = 0;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT: { running = 0; } break;
            case SDL_MOUSEMOTION: {
                mouse_x += event.motion.xrel;
                mouse_y += event.motion.yrel;
            } break;
            case SDL_KEYDOWN: {
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE: { running = false; } break;
                case SDLK_w:      { keymap.w = 1; } break;
                case SDLK_s:      { keymap.s = 1; } break;
                case SDLK_a:      { keymap.a = 1; } break;
                case SDLK_d:      { keymap.d = 1; } break;
                case SDLK_UP:     { keymap.up = 1; } break;
                case SDLK_DOWN:   { keymap.down = 1; } break;
                case SDLK_LEFT:   { keymap.left = 1; } break;
                case SDLK_RIGHT:  { keymap.right = 1; } break;
                }
            } break;
            case SDL_KEYUP: {
                switch (event.key.keysym.sym) {
                case SDLK_w:     { keymap.w = 0; } break;
                case SDLK_s:     { keymap.s = 0; } break;
                case SDLK_a:     { keymap.a = 0; } break;
                case SDLK_d:     { keymap.d = 0; } break;
                case SDLK_UP:    { keymap.up = 0; } break;
                case SDLK_DOWN:  { keymap.down = 0; } break;
                case SDLK_LEFT:  { keymap.left = 0; } break;
                case SDLK_RIGHT: { keymap.right = 0; } break;
                }
            } break;
            }
        }

        if (keymap.w) { camera.pos = camera.pos + seconds * camera.w; acc_frame = 0; }
        if (keymap.s) { camera.pos = camera.pos - seconds * camera.w; acc_frame = 0; }
        if (keymap.a) { camera.pos = camera.pos - seconds * camera.u; acc_frame = 0; }
        if (keymap.d) { camera.pos = camera.pos + seconds * camera.u; acc_frame = 0; }

        if (keymap.up)    { camera.inclination -= seconds; acc_frame = 0; }
        if (keymap.down)  { camera.inclination += seconds; acc_frame = 0; }
        if (keymap.left)  { camera.azimuth -= seconds; acc_frame = 0; }
        if (keymap.right) { camera.azimuth += seconds; acc_frame = 0; }

        if (mouse_x != 0 || mouse_y != 0) { acc_frame = 0; }

        camera.inclination += 0.005f * mouse_y;
        camera.azimuth += 0.005f * mouse_x;

        camera.update_uvw();

        render(gpu_bvh, gpu_triangles, camera, skybox_buffer, 
               frame_buffer, SCREEN_WIDTH, SCREEN_HEIGHT, cast(u32, frame_count));
        accumulate(frame_buffer, accumulator, screen_buffer, SCREEN_WIDTH, SCREEN_HEIGHT, acc_frame);
        render_buffer_to_screen(screen_array, screen_buffer, SCREEN_WIDTH, SCREEN_HEIGHT);
        //render_buffer_to_screen(screen_array, frame_buffer, SCREEN_WIDTH, SCREEN_HEIGHT);

        glBlitNamedFramebuffer(gl_frame_buffer, 
                               0, 0, 0, 
                               SCREEN_WIDTH, SCREEN_HEIGHT, 0, 
                               SCREEN_HEIGHT, SCREEN_WIDTH, 0, 
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        SDL_GL_SwapWindow(window);

        frame_count++;
        acc_frame++;
    }

    cudaGraphicsUnregisterResource(graphics_resource);
    glDeleteRenderbuffers(1, &gl_render_buffer);
    glDeleteFramebuffers(1, &gl_frame_buffer);

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}

// The code below is another way of achieving CUDA/OpenGL interop, which uses
// OpenGL extensions of a lower OpenGL version. The above version is a little
// bit faster (at least for me) but makes use of extensions of OpenGL version
// 4.5 or something. So this is a fall back in case you don't have that
// version. 
#if 0
int main_new(int, char**);
int main_simple(int, char**);

int main(int argc, char** args) {
    return main_new(argc, args);
}

int main_simple(int argc, char** args) {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 0);
    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_Window* window = SDL_CreateWindow("sting",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          SCREEN_WIDTH,
                                          SCREEN_HEIGHT,
                                          SDL_WINDOW_OPENGL);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);

    // disable the vsync for max fps
    SDL_GL_SetSwapInterval(0);

    if (!load_gl_extensions()) {
        return 1;
    }

    // ------------------------------------------------------------------------

    const char* vertex_shader_source =
        "#version 330\n"
        "layout (location = 0) in vec2 pos;\n"
        "layout (location = 1) in vec2 in_uv;\n"
        "out vec2 uv;\n"
        "void main() {\n"
        "    gl_Position = vec4(pos.x, pos.y, 0, 1);\n"
        "    uv = in_uv;\n"
        "}\n";

    const char* fragment_shader_source =
        "#version 330\n"
        "in vec2 uv;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "    float v = texture(tex, uv).r;\n"
        "    gl_FragColor = texture(tex, uv);\n"
        "}\n";

    // position and uv information for 6 vertices
    const float vertices[] = {
        -1.0f,  1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 1.0f,

        -1.0f,  1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 1.0f,
    };

    GLint status;
    char error_log[512];

    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);

    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        glGetShaderInfoLog(vertex_shader, 512, NULL, error_log);
        printf("Failed to compile the vertex shader\n");
        return 1;
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        glGetShaderInfoLog(fragment_shader, 512, NULL, error_log);
        printf("Failed to compile the fragment shader\n");
        return 1;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        glGetProgramInfoLog(program, 512, NULL, error_log);
        return 1;
    }

    glUseProgram(program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint gl_screen_texture;
    glGenTextures(1, &gl_screen_texture);
    glBindTexture(GL_TEXTURE_2D, gl_screen_texture);
    glActiveTexture(GL_TEXTURE0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);

    cudaGraphicsResource* cuda_screen_texture;
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&cuda_screen_texture, 
                                           gl_screen_texture, GL_TEXTURE_2D, 
                                           cudaGraphicsRegisterFlagsWriteDiscard));

    cudaArray* cuda_screen_array;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_screen_texture, 0));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_screen_array, cuda_screen_texture, 0, 0));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_screen_texture, 0));

    // In CUDA write to this buffer, we will map this to the OpenGL texture
    vec4* cuda_screen_buffer;
    CUDA_CHECK(cudaMalloc(&cuda_screen_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4)));

    // ------------------------------------------------------------------------

    PointCamera camera(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, -1), SCREEN_WIDTH, SCREEN_HEIGHT, 600);
    Keymap keymap = Keymap::empty();

    int running = 1;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT: { running = 0; } break;
            case SDL_KEYDOWN: {
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE: { running = 0; } break;
                case SDLK_w: { keymap.w = 1; } break;
                case SDLK_s: { keymap.s = 1; } break;
                case SDLK_a: { keymap.a = 1; } break;
                case SDLK_d: { keymap.d = 1; } break;
                case SDLK_UP: { keymap.up = 1; } break;
                case SDLK_DOWN: { keymap.down = 1; } break;
                case SDLK_LEFT: { keymap.left = 1; } break;
                case SDLK_RIGHT: { keymap.right = 1; } break;
                }
            } break;
            case SDL_KEYUP: {
                switch (event.key.keysym.sym) {
                case SDLK_w: { keymap.w = 0; } break;
                case SDLK_s: { keymap.s = 0; } break;
                case SDLK_a: { keymap.a = 0; } break;
                case SDLK_d: { keymap.d = 0; } break;
                case SDLK_UP: { keymap.up = 0; } break;
                case SDLK_DOWN: { keymap.down = 0; } break;
                case SDLK_LEFT: { keymap.left = 0; } break;
                case SDLK_RIGHT: { keymap.right = 0; } break;
                }
            } break;
            }
        }

        if (keymap.w) { camera.pos = camera.pos + camera.w; }
        if (keymap.s) { camera.pos = camera.pos - camera.w; }
        if (keymap.a) { camera.pos = camera.pos - camera.u; }
        if (keymap.d) { camera.pos = camera.pos + camera.u; }

        if (keymap.up)   { camera.inclination -= 0.001f; }
        if (keymap.down) { camera.inclination += 0.001f; }

        if (keymap.left)  { camera.azimuth -= 0.001f; }
        if (keymap.right) { camera.azimuth += 0.001f; }

        camera.update_uvw();

        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        CUDA_CHECK(cudaMemcpy2DToArray(cuda_screen_array, 0, 0, cuda_screen_buffer,
                                       SCREEN_WIDTH * sizeof(vec4), SCREEN_WIDTH * sizeof(vec4), SCREEN_HEIGHT,
                                       cudaMemcpyDeviceToDevice));

        //fill_buffer(cuda_screen_buffer, camera, SCREEN_WIDTH, SCREEN_HEIGHT);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
#endif

