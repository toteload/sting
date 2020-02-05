#include "ext/SDL2-2.0.10/include/SDL.h"
#include <stdio.h>
#include <stdint.h>
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <Windows.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "common.h"
#include "glext.h"
#include "load_gl_extensions.h"
#include "vecmath.h"
#include "bvh.h"
#include "bvh.cpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"
#pragma clang diagnostic pop

const uint32_t STING_VERSION_MAJOR = 0;
const uint32_t STING_VERSION_MINOR = 1;
const uint32_t STING_VERSION_REVISION = 0;

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

void render(BVHNode const * bvh, RenderTriangle const * triangles, PointCamera camera, 
            vec4* screen_buffer, uint32_t width, uint32_t height);
void render_buffer_to_screen(cudaArray_const_t array, vec4* buffer, uint32_t width, uint32_t height);

int main(int argc, char** args) {
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

    // enable/disable vsync
    SDL_GL_SetSwapInterval(0);

    // Load all the OpenGL extension functions
    // --------------------------------------------------------------------- //
    PFNGLCREATERENDERBUFFERSPROC glCreateRenderbuffers;
    PFNGLCREATEFRAMEBUFFERSPROC glCreateFramebuffers;
    PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC glNamedFramebufferRenderbuffer;
    PFNGLNAMEDRENDERBUFFERSTORAGEPROC glNamedRenderbufferStorage;
    PFNGLBLITNAMEDFRAMEBUFFERPROC glBlitNamedFramebuffer;
    PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
    PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;

    glCreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glCreateRenderbuffers");
    glCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glCreateFramebuffers");
    glNamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC)SDL_GL_GetProcAddress("glNamedFramebufferRenderbuffer");
    glNamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC)SDL_GL_GetProcAddress("glNamedRenderbufferStorage");
    glBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC)SDL_GL_GetProcAddress("glBlitNamedFramebuffer");
    glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteRenderbuffers");
    glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteFramebuffers");

    if (!glCreateRenderbuffers) { printf("Could not find glCreateRenderbuffers...\n"); return 1; }
    if (!glCreateFramebuffers) { printf("Could not find glCreateFramebuffers...\n"); return 1; }
    if (!glNamedFramebufferRenderbuffer) { printf("Could not find glNamedFramebufferRenderbuffer...\n"); return 1; }
    if (!glNamedRenderbufferStorage) { return 1; }
    if (!glBlitNamedFramebuffer) { return 1; }
    if (!glDeleteRenderbuffers) { return 1; }
    if (!glDeleteFramebuffers) { return 1; }

    GLuint framebuffer, renderbuffer;
    cudaArray* screen_array;
    cudaGraphicsResource* graphics_resource;

    glCreateRenderbuffers(1, &renderbuffer);
    glCreateFramebuffers(1, &framebuffer);
    glNamedFramebufferRenderbuffer(framebuffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
    glNamedRenderbufferStorage(renderbuffer, GL_RGBA32F, SCREEN_WIDTH, SCREEN_HEIGHT);

    cudaGraphicsGLRegisterImage(&graphics_resource,
                                renderbuffer, GL_RENDERBUFFER, 
                                cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &graphics_resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&screen_array, graphics_resource, 0, 0);
    cudaGraphicsUnmapResources(1, &graphics_resource, 0);

    vec4* screen_buffer;
    cudaMalloc(&screen_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(vec4));

    PointCamera camera(vec3(-1, 1, 0), // position
                       vec3(0, 1, 0),  // up
                       vec3(0, 0, 0),  // at
                       SCREEN_WIDTH/1.0f, SCREEN_HEIGHT/1.0f, 600);
    Keymap keymap = Keymap::empty();

    fastObjMesh* mesh = fast_obj_read("Thai_Buddha.obj");

    if (!mesh) {
        printf("Failed to load Buddha...\n");
        return 1;
    }

    std::vector<RenderTriangle> triangles;

    // Loop over all the groups in the mesh
    for (uint32_t i = 0; i < mesh->group_count; i++) {
        fastObjGroup group = mesh->groups[i];

        uint32_t vertex_index = 0;

        // Loop over all the faces in this group
        for (uint32_t j = 0; j < group.face_count; j++) {
            const uint32_t vertex_count = mesh->face_vertices[group.face_offset + j];

            if (vertex_count != 3) {
                // TODO report some sort of warning or error
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

            triangles.push_back(RenderTriangle(vertices[0], vertices[1], vertices[2]));
        }
    }

    printf("%llu RenderTriangles created...\n", triangles.size());
    
    std::vector<BVHNode> bvh = build_bvh_for_triangles(triangles.data(), triangles.size());

    printf("Created %llu bvh nodes...\n", bvh.size());

    fast_obj_destroy(mesh);

    RenderTriangle* gpu_triangles;
    CUDA_CHECK(cudaMalloc(&gpu_triangles, triangles.size() * sizeof(RenderTriangle)));

    BVHNode* gpu_bvh;
    CUDA_CHECK(cudaMalloc(&gpu_bvh, bvh.size() * sizeof(BVHNode)));

    CUDA_CHECK(cudaMemcpy(gpu_triangles, triangles.data(), triangles.size() * sizeof(RenderTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_bvh, bvh.data(), bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    uint64_t frame_count = 0;

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

        if (keymap.w) { camera.pos = camera.pos + 0.01f * camera.w; }
        if (keymap.s) { camera.pos = camera.pos - 0.01f * camera.w; }
        if (keymap.a) { camera.pos = camera.pos - 0.01f * camera.u; }
        if (keymap.d) { camera.pos = camera.pos + 0.01f * camera.u; }

        if (keymap.up)   { camera.inclination -= 0.01f; }
        if (keymap.down) { camera.inclination += 0.01f; }

        if (keymap.left)  { camera.azimuth -= 0.01f; }
        if (keymap.right) { camera.azimuth += 0.01f; }

        camera.update_uvw();

        render(gpu_bvh, gpu_triangles, camera, screen_buffer, SCREEN_WIDTH, SCREEN_HEIGHT);

        render_buffer_to_screen(screen_array, screen_buffer, SCREEN_WIDTH, SCREEN_HEIGHT);

        glBlitNamedFramebuffer(framebuffer, 
                               0, 0, 0, 
                               SCREEN_WIDTH, SCREEN_HEIGHT, 0, 
                               SCREEN_HEIGHT, SCREEN_WIDTH, 0, 
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        SDL_GL_SwapWindow(window);

        frame_count++;
    }

    cudaGraphicsUnregisterResource(graphics_resource);
    glDeleteRenderbuffers(1, &renderbuffer);
    glDeleteFramebuffers(1, &framebuffer);

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}

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

