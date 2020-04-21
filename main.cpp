#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <Windows.h>
#include "SDL.h"
#include <GL/gl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudaGL.h>

#include <stdint.h>
#include <stdio.h>

#include "dab/dab.h"
#include "ext/glext.h"
#include "ext/fast_obj.h"
#include "ext/stb_image.h"
#include "porky_load.cpp"
#include "stingmath.h"
#include "camera.h"
#include "bvh.h"
#include "bvh.cpp"

#include "wavefront.h"

#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM "porky_load.h"
#include "imgui/imconfig.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_opengl3.h"

const uint32_t STING_VERSION_MAJOR = 0;
const uint32_t STING_VERSION_MINOR = 1;
const uint32_t STING_VERSION_REVISION = 1;

inline CUresult cuda_err_check(CUresult err, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* cu_err_name;
        const char* cu_err_string;
        cuGetErrorName(err, &cu_err_name);
        cuGetErrorString(err, &cu_err_string);
        fprintf(stderr, "[cuda check failed] error code: %d, name: %s, message: %s, file: %s, line: %d\n",
                err, cu_err_name, cu_err_string, file, line);
    }

    return err;
}

#define CUDA_CHECK(...) cuda_err_check(__VA_ARGS__, __FILE__, __LINE__)

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

std::vector<RenderTriangle> generate_sphere_mesh(u32 rows, u32 columns, f32 radius, Vector3 pos) {
    if (rows < 2 || columns < 3) {
        return { };
    }

    std::vector<Vector3> pts;
    std::vector<Vector3> normals;

    for (u32 i = 1; i < rows; i++) {
        for (u32 j = 0; j < columns; j++) {
            const f32 phi = i * M_PI / cast(f32, rows);
            const f32 theta = (cast(f32, j) / cast(f32, columns)) * 2.0f * M_PI;
            const Vector3 n = spherical_to_cartesian(phi, theta);
            pts.push_back(pos + radius * n);
            normals.push_back(n);
        }
    }

    const size_t top = pts.size();
    pts.push_back(pos + radius * vec3(0.0f, 1.0f, 0.0f));
    normals.push_back(vec3(0.0f, 1.0f, 0.0f));

    const size_t bottom = pts.size();
    pts.push_back(pos + radius * vec3(0.0f, -1.0f, 0.0f));
    normals.push_back(vec3(0.0f, -1.0f, 0.0f));

    std::vector<RenderTriangle> triangles;

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;
#if 0
        triangles.push_back(RenderTriangle(    pts[top],     pts[i],     pts[inext],
                                           normals[top], normals[i], normals[inext]));
#endif
        triangles.push_back(RenderTriangle(pts[top], pts[i], pts[inext], 0));
    }

    for (u32 r = 0; r < rows - 2; r++) {
        for (u32 c = 0; c < columns; c++) {
            const u32 cnext = (c + 1) % columns;

            const u32 rowi = r * columns;
            const u32 rowinext = (r + 1) * columns;

#if 0
            triangles.push_back(RenderTriangle(    pts[rowi + c],     pts[rowinext + c],     pts[rowi + cnext],
                                               normals[rowi + c], normals[rowinext + c], normals[rowi + cnext]));
            triangles.push_back(RenderTriangle(    pts[rowinext + c],     pts[rowinext + cnext],     pts[rowi + cnext],
                                               normals[rowinext + c], normals[rowinext + cnext], normals[rowi + cnext]));
#endif
            triangles.push_back(RenderTriangle(pts[rowi + c], pts[rowinext + c], pts[rowi + cnext], 0));
            triangles.push_back(RenderTriangle(pts[rowinext + c], pts[rowinext + cnext], pts[rowi + cnext], 0));
        }
    }

    for (u32 i = 0; i < columns; i++) {
        const u32 inext = (i + 1) % columns;
        const u32 rowi = (rows - 2) * columns;

        triangles.push_back(RenderTriangle(pts[bottom], pts[rowi + inext], pts[rowi + i], 0));
#if 0
        triangles.push_back(RenderTriangle(    pts[bottom],     pts[rowi + inext],     pts[rowi + i],
                                           normals[bottom], normals[rowi + inext], normals[rowi + i]));
#endif
    }

    return triangles;
}

template<typename T>
void unused(const T& x) {
    (void)x;
}

struct Settings {
    u32 window_width, window_height;
    u32 buffer_width, buffer_height;
};

int main(int argc, char** args) {
    unused(argc); unused(args);

    Settings settings = {
        .window_width  = 1920/2,
        .window_height = 1080/2,
        .buffer_width  = 1920/4,
        .buffer_height = 1080/4,
    };

    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_FRAMEBUFFER_SRGB_CAPABLE, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    char name_buffer[128];
    snprintf(name_buffer, 128, "Sting %d.%d.%d", STING_VERSION_MAJOR, STING_VERSION_MINOR, STING_VERSION_REVISION);

    SDL_Window* window = SDL_CreateWindow(name_buffer,
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          settings.window_width,
                                          settings.window_height,
                                          SDL_WINDOW_OPENGL);// | SDL_WINDOW_FULLSCREEN_DESKTOP);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) {
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);

    if (!porky_load_extensions()) {
        return 1;
    }

    // enable/disable vsync
    SDL_GL_SetSwapInterval(0);

    // --------------------------------------------------------------------- //

    CUDA_CHECK(cuInit(0));

    CUdevice device;
    CUcontext context;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    CUmodule fillimage, wavefrontptx;
    CUDA_CHECK(cuModuleLoad(&fillimage, "fillimage.ptx"));
    CUDA_CHECK(cuModuleLoad(&wavefrontptx, "wavefront.ptx"));

    CUfunction render_bruteforce, render_nee, accumulate, blit_to_screen;
    CUDA_CHECK(cuModuleGetFunction(&render_nee, fillimage, "nee_test"));
    CUDA_CHECK(cuModuleGetFunction(&render_bruteforce, fillimage, "test_001"));
    CUDA_CHECK(cuModuleGetFunction(&accumulate, fillimage, "accumulate_pass"));
    CUDA_CHECK(cuModuleGetFunction(&blit_to_screen, fillimage, "blit_to_screen"));

    CUfunction reset, generate_primary_rays, extend_rays, shade;
    CUDA_CHECK(cuModuleGetFunction(&reset, wavefrontptx, "reset"));
    CUDA_CHECK(cuModuleGetFunction(&generate_primary_rays, wavefrontptx, "generate_primary_rays"));
    CUDA_CHECK(cuModuleGetFunction(&extend_rays, wavefrontptx, "extend_rays"));
    CUDA_CHECK(cuModuleGetFunction(&shade, wavefrontptx, "shade"));

    CUsurfref screen_surface;
    cuModuleGetSurfRef(&screen_surface, fillimage, "screen_surface");

    // CUDA/OpenGL interop setup
    // --------------------------------------------------------------------- //
    GLuint gl_frame_buffer, gl_render_buffer;
    glCreateRenderbuffers(1, &gl_render_buffer);
    glCreateFramebuffers(1, &gl_frame_buffer);
    glNamedFramebufferRenderbuffer(gl_frame_buffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, gl_render_buffer);
    glNamedRenderbufferStorage(gl_render_buffer, GL_RGBA32F, settings.buffer_width, settings.buffer_height);

    CUgraphicsResource graphics_resource;
    cuGraphicsGLRegisterImage(&graphics_resource, gl_render_buffer, GL_RENDERBUFFER, 
                              CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST | CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
    cuGraphicsMapResources(1, &graphics_resource, 0);

    CUarray screen_array;
    cuGraphicsSubResourceGetMappedArray(&screen_array, graphics_resource, 0, 0);
    cuGraphicsUnmapResources(1, &graphics_resource, 0);

    cuSurfRefSetArray(screen_surface, screen_array, 0);

    CUdeviceptr screen_buffer, frame_buffer, accumulator;
    CUDA_CHECK(cuMemAlloc(&screen_buffer, settings.buffer_width * settings.buffer_height * sizeof(Vector4)));
    CUDA_CHECK(cuMemAlloc(&frame_buffer,  settings.buffer_width * settings.buffer_height * sizeof(Vector4)));
    CUDA_CHECK(cuMemAlloc(&accumulator,   settings.buffer_width * settings.buffer_height * sizeof(Vector4)));

    // Wavefront state
    // --------------------------------------------------------------------- //
    CUdeviceptr wavefront_state, pathstate_buffer[2];
    CUDA_CHECK(cuMemAlloc(&wavefront_state, sizeof(wavefront::State)));
    CUDA_CHECK(cuMemAlloc(&pathstate_buffer[0], settings.buffer_width * settings.buffer_height * sizeof(wavefront::PathState)));
    CUDA_CHECK(cuMemAlloc(&pathstate_buffer[1], settings.buffer_width * settings.buffer_height * sizeof(wavefront::PathState)));

    // Loading in triangle mesh, building bvh and uploading to GPU
    // --------------------------------------------------------------------- //

    std::vector<RenderTriangle> triangles;

#if 0
    auto sphere_mesh = generate_sphere_mesh(36, 36, 0.2f, Vector3(0.0f, 0.0f, 0.0f));
    for (u32 i = 0; i < sphere_mesh.size(); i++) {
        sphere_mesh[i].material = MATERIAL_DIFFUSE;
    }
    triangles.insert(triangles.end(), sphere_mesh.begin(), sphere_mesh.end());

    auto spheremesh0 = generate_sphere_mesh(36, 36, 0.3f, Vector3(0.4f, 0.0f, 0.0f));
    for (u32 i = 0; i < spheremesh0.size(); i++) {
        spheremesh0[i].material = MATERIAL_DIFFUSE;
    }
    triangles.insert(triangles.end(), spheremesh0.begin(), spheremesh0.end());
#endif

#if 0
    auto light0 = RenderTriangle(Vector3(0.0f, 0.4f, 0.0f),
                                 Vector3(0.3f, 0.4f, 0.0f),
                                 Vector3(0.0f, 0.5f, 0.3f));
    light0.material = MATERIAL_EMISSIVE;
    light0.colorr = light0.colorg = 0.0f;
    light0.light_intensity = 1.0f;
    triangles.push_back(light0);

    auto light1 = RenderTriangle(Vector3(0.0f, -0.4f, 0.0f),
                                 Vector3(0.3f, -0.4f, 0.0f),
                                 Vector3(0.0f, -0.5f, 0.3f));
    light1.material = MATERIAL_EMISSIVE;
    light1.colorb = light1.colorg = 0.0f;
    light1.light_intensity = 1.0f;
    triangles.push_back(light1);

    auto light2 = RenderTriangle(Vector3(0.0f, 0.0f, -0.8f),
                                 Vector3(0.3f, 0.0f, -0.8f),
                                 Vector3(0.3f, 0.5f, -0.8f));
    light2.material = MATERIAL_EMISSIVE;
    light2.colorb = light2.colorr = 0.0f;
    light2.light_intensity = 4.0f;
    triangles.push_back(light2);
#endif

#if 1
    //fastObjMesh* mesh = fast_obj_read("Thai_Buddha.obj");
    fastObjMesh* mesh = fast_obj_read("data/sponza.obj");

    printf("Mesh has %d vertices, and %d normals\n", mesh->position_count, mesh->normal_count);

    std::vector<Material> materials;
    materials.push_back({ .type = Material::DIFFUSE, .r = 1.0f, .g = 1.0f, .b = 1.0f, });

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

            Vector3 vertices[3];

            // Loop over all the vertices in this face
            for (uint32_t k = 0; k < vertex_count; k++) {
                fastObjIndex index = mesh->indices[group.index_offset + vertex_index];

                vertices[k] = vec3(mesh->positions[3 * index.p + 0], 
                                   mesh->positions[3 * index.p + 1], 
                                   mesh->positions[3 * index.p + 2]);

                vertex_index++;
            }
            auto tri = RenderTriangle(vertices[0], vertices[1], vertices[2], 0);
            triangles.push_back(tri);
        }
    }

    fast_obj_destroy(mesh);
#endif

    printf("%llu RenderTriangles created...\n", triangles.size());
    
    u32 bvh_depth, max_primitives;
    std::vector<BVHNode> bvh = build_bvh_for_triangles(triangles.data(), triangles.size(), &bvh_depth, &max_primitives);

#if 0
    std::vector<u32> lights;
    for (u32 i = 0; i < triangles.size(); i++) {
        if (triangles[i].material == MATERIAL_EMISSIVE) {
            lights.push_back(i);
            printf("light found at %d\n", i);
        }
    }
#endif

    printf("Created %llu bvh nodes...\n", bvh.size());
    printf("BVH depth %d, max triangles in leaf %d\n", bvh_depth, max_primitives);

    CUdeviceptr gpu_triangles, gpu_bvh, gpu_materials;//, gpu_lights;
    CUDA_CHECK(cuMemAlloc(&gpu_triangles, triangles.size() * sizeof(RenderTriangle)));
    CUDA_CHECK(cuMemAlloc(&gpu_bvh, bvh.size() * sizeof(BVHNode)));
    CUDA_CHECK(cuMemAlloc(&gpu_materials, materials.size() * sizeof(Material)));
    //cuMemAlloc(&gpu_lights, lights.size() * sizeof(u32));

    CUDA_CHECK(cuMemcpyHtoD(gpu_triangles, triangles.data(), triangles.size() * sizeof(RenderTriangle)));
    CUDA_CHECK(cuMemcpyHtoD(gpu_bvh, bvh.data(), bvh.size() * sizeof(BVHNode)));
    CUDA_CHECK(cuMemcpyHtoD(gpu_materials, materials.data(), materials.size() * sizeof(Material)));
    //cuMemcpyHtoD(gpu_lights, lights.data(), lights.size() * sizeof(u32));

    // Skybox loading
    // --------------------------------------------------------------------- //
#if 1
    i32 hdr_width, hdr_height, hdr_channels;
    u8* skybox_image = stbi_load("data/cloudysky_4k.jpg", &hdr_width, &hdr_height, &hdr_channels, 4);
    if (!skybox_image) {
        printf("Failed to load sky dome...\n");
    }

    printf("Loaded skybox texture, width: %d, height: %d\n", hdr_width, hdr_height);

    Vector4* skybox = cast(Vector4*, malloc(hdr_width * hdr_height * sizeof(Vector4)));
    for (i32 i = 0; i < hdr_width * hdr_height; i++) {
        skybox[i] = vec4(skybox_image[i * 4 + 0] / 255.0f,
                         skybox_image[i * 4 + 1] / 255.0f,
                         skybox_image[i * 4 + 2] / 255.0f,
                         1.0f);
    }

    CUdeviceptr skybox_buffer;
    CUDA_CHECK(cuMemAlloc(&skybox_buffer, hdr_width * hdr_height * sizeof(Vector4)));
    CUDA_CHECK(cuMemcpyHtoD(skybox_buffer, skybox, hdr_width * hdr_height * sizeof(Vector4)));

    free(skybox);
    stbi_image_free(skybox_image);
#endif

    // Setting the camera and keymap
    // --------------------------------------------------------------------- //
    PointCamera camera(vec3(-0.5f, 0.5f, 0.5f), // position
                       vec3(0, 1, 0), // up
                       vec3(0, 0, 0), // at
                       settings.window_width, settings.window_height, 
                       600);

    Keymap keymap = Keymap::empty();

    SDL_SetRelativeMouseMode(SDL_TRUE);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiIO& io = ImGui::GetIO();
    
    if (!ImGui_ImplSDL2_InitForOpenGL(window, gl_context)) {
        printf("Failed to initiazlie ImGui for SDL...\n");
        return 1;
    }

    if (!ImGui_ImplOpenGL3_Init(NULL)) {
        printf("Failed to initialize ImGui for OpenGL...\n");
        return 1; 
    }

    uint64_t frame_count = 0;
    u32 acc_frame = 0;

    const u64 frequency = SDL_GetPerformanceFrequency();
    u64 previous_counter = SDL_GetPerformanceCounter();

    u32 rendermethod = 0;
    u32 accumulate_toggle = 1;

    bool show_demo = false;
    bool show_controls = false;

    f32 frame_time = 0;

    glEnable(GL_FRAMEBUFFER_SRGB);

    bool running = true;
    while (running) {
        const u64 current_counter = SDL_GetPerformanceCounter();
        const u64 time_diff = current_counter - previous_counter;
        const f32 seconds = cast(f32, time_diff) / frequency;
        previous_counter = current_counter;

        i32 mouse_x = 0, mouse_y = 0;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);

            switch (event.type) {
            case SDL_QUIT: { running = 0; } break;
#if 1
            case SDL_MOUSEBUTTONDOWN: {
                if (io.WantCaptureMouse) { break; }

                if (event.button.button == SDL_BUTTON_LEFT) {
                    if (!SDL_GetRelativeMouseMode()) {
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                    }
                }
            } break;
            case SDL_MOUSEMOTION: {
                if (io.WantCaptureMouse) { break; }

                mouse_x += event.motion.xrel;
                mouse_y += event.motion.yrel;
            } break;
            case SDL_KEYDOWN: {
                if (io.WantCaptureKeyboard) { break; }

                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE: { 
                    if (!SDL_GetRelativeMouseMode()) {
                        running = false; 
                    } else {
                        SDL_SetRelativeMouseMode(SDL_FALSE);
                        SDL_WarpMouseInWindow(window, settings.window_width / 2, settings.window_height / 2);
                    }
                } break;

                case SDLK_w:      { keymap.w = 1; } break;
                case SDLK_s:      { keymap.s = 1; } break;
                case SDLK_a:      { keymap.a = 1; } break;
                case SDLK_d:      { keymap.d = 1; } break;
                case SDLK_UP:     { keymap.up = 1; } break;
                case SDLK_DOWN:   { keymap.down = 1; } break;
                case SDLK_LEFT:   { keymap.left = 1; } break;
                case SDLK_RIGHT:  { keymap.right = 1; } break;
                case SDLK_1:      { rendermethod ^= 1; acc_frame = 0; } break;
                case SDLK_2:      { accumulate_toggle ^= 1; acc_frame = 0; } break;
                }
            } break;
            case SDL_KEYUP: {
                if (io.WantCaptureKeyboard) { break; }

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
#endif
            }
        }

        if (keymap.w) { camera.pos = camera.pos + seconds * 1000 * camera.w; acc_frame = 0; }
        if (keymap.s) { camera.pos = camera.pos - seconds * 1000 * camera.w; acc_frame = 0; }
        if (keymap.a) { camera.pos = camera.pos - seconds * 1000 * camera.u; acc_frame = 0; }
        if (keymap.d) { camera.pos = camera.pos + seconds * 1000 * camera.u; acc_frame = 0; }

        if (keymap.up)    { camera.inclination -= seconds; acc_frame = 0; }
        if (keymap.down)  { camera.inclination += seconds; acc_frame = 0; }
        if (keymap.left)  { camera.azimuth -= seconds; acc_frame = 0; }
        if (keymap.right) { camera.azimuth += seconds; acc_frame = 0; }

        if (SDL_GetRelativeMouseMode()) {
            if (mouse_x != 0 || mouse_y != 0) { acc_frame = 0; }

            camera.inclination += 0.005f * mouse_y;
            camera.azimuth     += 0.005f * mouse_x;

            camera.update_uvw();
        }

#if 0
        const u32 block_x = 16, block_y = 16;
        const u32 grid_x = (settings.buffer_width + block_x - 1) / block_x; 
        const u32 grid_y = (settings.buffer_height + block_y - 1) / block_y;

        //u32 light_count = lights.size();

#if 1
        void* render_params[] = {
            &gpu_bvh, 
            &gpu_triangles, 
            &gpu_materials,
            &camera,
            &skybox_buffer,
            &frame_buffer, 
            &settings.buffer_width, 
            &settings.buffer_height, 
            &frame_count,
        };

        CUDA_CHECK(cuLaunchKernel(render_bruteforce, 
                                  grid_x, grid_y, 1, 
                                  block_x, block_y, 1, 
                                  0, 
                                  0, 
                                  render_params, 
                                  NULL));
#endif

#if 0
        void* render_params[] = {
            &gpu_bvh, &gpu_triangles, &gpu_lights, &light_count, &camera,
            &skybox_buffer, &frame_buffer, &width, &height, &frame_count,
        };
        cuLaunchKernel(render_nee, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, render_params, NULL);
#endif

#if 1
        void* accumulate_params[] = {
            &frame_buffer, &accumulator, &screen_buffer, &settings.buffer_width, &settings.buffer_height, &acc_frame,
        };
        cuLaunchKernel(accumulate, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, accumulate_params, NULL);
#endif

        void* blit_params[] = {
            &screen_buffer, 
            &settings.buffer_width, 
            &settings.buffer_height,
        };
        CUDA_CHECK(cuLaunchKernel(blit_to_screen, 
                                  grid_x, grid_y, 1, 
                                  block_x, block_y, 1, 
                                  0, 
                                  0, 
                                  blit_params, 
                                  NULL));
#else
        wavefront::State wavefront_state_values;
        wavefront_state_values.job_count[0] = 0;
        wavefront_state_values.job_count[1] = 0;
        wavefront_state_values.states[0] = cast(wavefront::PathState*, pathstate_buffer[0]);
        wavefront_state_values.states[1] = cast(wavefront::PathState*, pathstate_buffer[1]);

        CUDA_CHECK(cuMemcpyHtoD(wavefront_state, &wavefront_state_values, sizeof(wavefront::State)));
        CUDA_CHECK(cuMemsetD8(frame_buffer, 0, settings.buffer_width * settings.buffer_height * sizeof(Vector4)));

        const u32 block_x = 128;
        const u32 grid_x = (settings.buffer_width * settings.buffer_height + block_x - 1) / block_x;

        u32 current = 0;

        {
            void* generate_primary_rays_params[] = {
                &wavefront_state, 
                &current, 
                &camera, 
                &settings.buffer_width, 
                &settings.buffer_height, 
                &frame_count,
            };

            CUDA_CHECK(cuLaunchKernel(generate_primary_rays, 
                                      grid_x, 1, 1, 
                                      block_x, 1, 1, 
                                      0, 
                                      0, 
                                      generate_primary_rays_params, 
                                      NULL));
        }

        for (u32 i = 0; i < 3; i++) {
            void* reset_params[] = {
                &wavefront_state,
                &current,
            };

            cuLaunchKernel(reset,
                           1, 1, 1,
                           1, 1, 1,
                           0,
                           0,
                           reset_params,
                           NULL);

            void* extend_rays_params[] = {
                &wavefront_state, 
                &current, 
                &gpu_bvh, 
                &gpu_triangles,
            };

            CUDA_CHECK(cuLaunchKernel(extend_rays, 
                                      grid_x, 1, 1, 
                                      block_x, 1, 1, 
                                      0, 
                                      0, 
                                      extend_rays_params, 
                                      NULL));

            void* shade_params[] = {
                &wavefront_state, 
                &current, 
                &gpu_triangles, 
                &gpu_materials,
                &frame_buffer,
            };

            CUDA_CHECK(cuLaunchKernel(shade, 
                                      grid_x, 1, 1, 
                                      block_x, 1, 1, 
                                      0, 
                                      0, 
                                      shade_params, 
                                      NULL));

            current ^= 1;
        }

        {
            const u32 block_x = 16, block_y = 16;
            const u32 grid_x = (settings.buffer_width + block_x - 1) / block_x; 
            const u32 grid_y = (settings.buffer_height + block_y - 1) / block_y;
            void* accumulate_params[] = {
                &frame_buffer, &accumulator, &screen_buffer, &settings.buffer_width, &settings.buffer_height, &acc_frame,
            };
            cuLaunchKernel(accumulate, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, accumulate_params, NULL);
        }

        {
            const u32 block_x = 16, block_y = 16;
            const u32 grid_x = (settings.buffer_width + block_x - 1) / block_x; 
            const u32 grid_y = (settings.buffer_height + block_y - 1) / block_y;
            void* blit_params[] = {
                &screen_buffer, &settings.buffer_width, &settings.buffer_height,
            };
            CUDA_CHECK(cuLaunchKernel(blit_to_screen, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, blit_params, NULL));
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#endif
        glBlitNamedFramebuffer(gl_frame_buffer, 0, 
                               0, 0, settings.buffer_width, settings.buffer_height, 
                               0, settings.window_height, settings.window_width, 0, 
                               GL_COLOR_BUFFER_BIT, GL_LINEAR);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("Options")) {
                if (ImGui::MenuItem("Show demo")) {
                    show_demo = true;
                }

                if (ImGui::MenuItem("Show controls")) {
                    show_controls = true;
                }

                if (ImGui::MenuItem("Quit")) {
                    running = false;
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            ImGui::Text("%8.3f ms/frame, fps: %6.1f", 1000 * frame_time, 1.0f / frame_time);

            ImGui::Separator();

            i32 mouse_x, mouse_y;
            SDL_GetMouseState(&mouse_x, &mouse_y);
            ImGui::Text("Mouse: %4d, %4d", mouse_x, mouse_y);

            ImGui::Separator();

            i32 window_width, window_height;
            SDL_GetWindowSize(window, &window_width, &window_height);
            ImGui::Text("Window: %4dx%4d, frame buffer: %4dx%4d", 
                        settings.window_width, settings.window_height, 
                        settings.buffer_width, settings.buffer_height);

            ImGui::EndMainMenuBar();
        }

        if (show_demo) { ImGui::ShowDemoWindow(&show_demo); }

        bool reset_frame = false;
        if (show_controls) {
            ImGui::Begin("Controls");

            if (ImGui::InputFloat3("Position", cast(f32*, &camera.pos), "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
                reset_frame = true;
            }

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);

        cuCtxSynchronize();

        const u64 end_counter = SDL_GetPerformanceCounter();
        frame_time = cast(f32, end_counter - current_counter) / frequency;

        frame_count++;
        acc_frame++;

        if (reset_frame) {
            acc_frame = 0;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    cuGraphicsUnregisterResource(graphics_resource);
    cuCtxDestroy(context);

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
        "    gl_Position = Vector4(pos.x, pos.y, 0, 1);\n"
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
    Vector4* cuda_screen_buffer;
    CUDA_CHECK(cudaMalloc(&cuda_screen_buffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Vector4)));

    // ------------------------------------------------------------------------

    PointCamera camera(Vector3(0, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, -1), SCREEN_WIDTH, SCREEN_HEIGHT, 600);
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
                                       SCREEN_WIDTH * sizeof(Vector4), SCREEN_WIDTH * sizeof(Vector4), SCREEN_HEIGHT,
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

