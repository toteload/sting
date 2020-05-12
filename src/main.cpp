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
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "dab/dab.h"
#include "ext/glext.h"
#include "ext/fast_obj.h"
#include "ext/stb_image.h"
#include "porky_load.cpp"
#include "stingmath.h"
#include "camera.h"
#include "mesh.cpp"
#include "bvh.h"
#include "bvh.cpp"

#include "wavefront.h"

#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM "porky_load.h"
#include "ext/imgui/imconfig.h"
#include "ext/imgui/imgui.h"
#include "ext/imgui/imgui_internal.h"
#include "ext/imgui/imgui_impl_sdl.h"
#include "ext/imgui/imgui_impl_opengl3.h"
#include "dragfloatprecise.cpp"

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

bool does_file_exist(const char* filename) {
    const DWORD attribs = GetFileAttributes(filename);
    return (attribs != INVALID_FILE_ATTRIBUTES) && !(attribs & FILE_ATTRIBUTE_DIRECTORY); 
}

struct Settings {
    u32 window_width, window_height;
    u32 buffer_width, buffer_height;
};

int main(int argc, char** args) {
    DAB_UNUSED(argc); 
    DAB_UNUSED(args);

    Settings settings = {
        .window_width  = cast(u32, 1920/1.5),
        .window_height = cast(u32, 1080/1.5),

        .buffer_width  = cast(u32, 1920/3.0),
        .buffer_height = cast(u32, 1080/3.0),
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

    i32 cuda_device_count = 0;
    CUDA_CHECK(cuDeviceGetCount(&cuda_device_count));

    CUdevice device;
    CUcontext context;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    i32 sm_count, max_threads_per_sm, max_threads_per_block, warp_size;
    CUDA_CHECK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CUDA_CHECK(cuDeviceGetAttribute(&max_threads_per_sm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
    CUDA_CHECK(cuDeviceGetAttribute(&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    CUDA_CHECK(cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));

    printf("SM count: %d\n"
           "max threads per SM: %d\n"
           "max threads per block: %d\n"
           "warp size: %d\n"
           , 
           sm_count, 
           max_threads_per_sm,
           max_threads_per_block,
           warp_size);

    CUmodule pathtrace_ptx, wavefront_ptx;
    CUDA_CHECK(cuModuleLoad(&pathtrace_ptx, "pathtrace.ptx"));
    CUDA_CHECK(cuModuleLoad(&wavefront_ptx, "wavefront.ptx"));

    CUfunction accumulate, blit_to_screen;
    CUDA_CHECK(cuModuleGetFunction(&accumulate,        pathtrace_ptx, "accumulate_pass"));
    CUDA_CHECK(cuModuleGetFunction(&blit_to_screen,    pathtrace_ptx, "blit_to_screen"));

    CUfunction reset, generate_primary_rays, extend_rays, extend_rays_compressed, shade, output_to_buffer;
    CUfunction extend_rays_compressed_fetch;
    CUDA_CHECK(cuModuleGetFunction(&reset,                  wavefront_ptx, "reset"));
    CUDA_CHECK(cuModuleGetFunction(&generate_primary_rays,  wavefront_ptx, "generate_primary_rays"));
    CUDA_CHECK(cuModuleGetFunction(&extend_rays,            wavefront_ptx, "extend_rays"));
    CUDA_CHECK(cuModuleGetFunction(&extend_rays_compressed, wavefront_ptx, "extend_rays_compressed"));
    CUDA_CHECK(cuModuleGetFunction(&output_to_buffer,       wavefront_ptx, "output_to_buffer"));
    CUDA_CHECK(cuModuleGetFunction(&shade,                  wavefront_ptx, "shade"));
    CUDA_CHECK(cuModuleGetFunction(&extend_rays_compressed_fetch, wavefront_ptx, "extend_rays_compressed_fetch"));

    CUsurfref screen_surface;
    cuModuleGetSurfRef(&screen_surface, pathtrace_ptx, "screen_surface");

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
    CUdeviceptr wavefront_state, pathstate_buffer, state_index[2];
    CUDA_CHECK(cuMemAlloc(&wavefront_state, sizeof(wavefront::State)));
    CUDA_CHECK(cuMemAlloc(&pathstate_buffer, settings.buffer_width * settings.buffer_height * sizeof(wavefront::PathState)));
    CUDA_CHECK(cuMemAlloc(&state_index[0], settings.buffer_width * settings.buffer_height * sizeof(u32)));
    CUDA_CHECK(cuMemAlloc(&state_index[1], settings.buffer_width * settings.buffer_height * sizeof(u32)));

    // Loading in triangle mesh, building bvh and uploading to GPU
    // --------------------------------------------------------------------- //
 
    std::vector<Vector4> triangles;
    std::vector<BVHNode> bvh;
    
#if 0
    // Create some spheres and save them to a file
    {
        for (u32 x = 0; x < 1; x++) {
            for (u32 z = 0; z < 1; z++) {
                for (u32 i = 0; i < 1; i++) {
                    const f32 theta = cast(f32, i) / 8.0f * 2.0f * M_PI;
                    triangles = generate_sphere_mesh(8, 8, 10.0f, vec3(x * 60 + 20.0f * sinf(theta),
                                                                       0.0f,
                                                                       z * 60 + 20.0f * cosf(theta)));
                }
            }
        }

        bvh = build_bvh_for_triangles_and_reorder(triangles, 12, 5);
        
        printf("%llu sphere triangles\n", triangles.size() / 3);
        
        save_bvh_object("spheres.bvh", bvh.data(), bvh.size(), triangles.data(), triangles.size());
    }
#endif
      
    const u64 frequency = SDL_GetPerformanceFrequency();

#if 1
    const char* bvh_filename = "sponza.bvh";

    if (!does_file_exist(bvh_filename)) {
        printf("Could not find bvh file. Rebuilding...\n");

        triangles = load_mesh_from_obj_file("data/sponza.obj");
        printf("%llu RenderTriangles created...\n", triangles.size() / 3);

        const u64 start_time = SDL_GetPerformanceCounter();
        bvh = build_bvh_for_triangles_and_reorder(triangles, 12, 5);
        const u64 end_time = SDL_GetPerformanceCounter();
     
        printf("Created %llu bvh nodes in %f ms...\n", 
               bvh.size(), 
               1000 * cast(f32, end_time - start_time) / frequency);

        save_bvh_object(bvh_filename, bvh.data(), bvh.size(), triangles.data(), triangles.size());
    } else {
        printf("Found bvh file. Loading from disk...\n");
        u64 bvh_file_size;
        void* bvh_data = read_file(bvh_filename, &bvh_file_size);
        printf("Loaded file of %llu bytes...\n", bvh_file_size);

        BVHObject bvh_object = load_bvh_object(bvh_data);
        if (!bvh_object.valid) {
            printf("Loaded bvh is not valid...\n");
            return 3;
        }

        bvh = bvh_object.bvh;
        triangles = bvh_object.triangles;

        free(bvh_data);
    }
#endif

#if 0
    build_mbvh8_for_triangles_and_reorder(triangles.data(), triangles.size());
    printf("done!\n");
    return 0;
#endif

    const u64 compression_start = SDL_GetPerformanceCounter();
    CBVH cbvh = compress_bvh(bvh);
    const u64 compression_end = SDL_GetPerformanceCounter();
    printf("Compressed bvh in %f ms...\n", 1000 * cast(f32, compression_end - compression_start) / frequency);
    
    CUdeviceptr gpu_triangles, gpu_bvh, gpu_cbvh;
    CUDA_CHECK(cuMemAlloc(&gpu_triangles, triangles.size()  * sizeof(Vector4)));
    CUDA_CHECK(cuMemAlloc(&gpu_bvh,       bvh.size()        * sizeof(BVHNode)));
    CUDA_CHECK(cuMemAlloc(&gpu_cbvh,      cbvh.nodes.size() * sizeof(CBVHNode)));

    CUDA_CHECK(cuMemcpyHtoD(gpu_triangles, triangles.data(),  triangles.size()  * sizeof(Vector4)));
    CUDA_CHECK(cuMemcpyHtoD(gpu_bvh,       bvh.data(),        bvh.size()        * sizeof(BVHNode)));
    CUDA_CHECK(cuMemcpyHtoD(gpu_cbvh,      cbvh.nodes.data(), cbvh.nodes.size() * sizeof(CBVHNode)));

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
    PointCamera camera(vec3(180, 140, -42), // position
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

    u64 previous_counter = SDL_GetPerformanceCounter();

    u32 rendermethod = 0;
    u32 accumulate_toggle = 1;

    bool show_demo = false;
    bool show_controls = true;

    f32 camera_speed = 100;
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

#if 1
        if (keymap.w) { camera.pos = camera.pos + seconds * camera_speed * camera.w; acc_frame = 0; }
        if (keymap.s) { camera.pos = camera.pos - seconds * camera_speed * camera.w; acc_frame = 0; }
        if (keymap.a) { camera.pos = camera.pos - seconds * camera_speed * camera.u; acc_frame = 0; }
        if (keymap.d) { camera.pos = camera.pos + seconds * camera_speed * camera.u; acc_frame = 0; }

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
#else
        if (frame_count == 40) { break; }
#endif

        wavefront::State wavefront_state_values;
        wavefront_state_values.total_ray_count = 0;
        wavefront_state_values.job_count[0] = 0;
        wavefront_state_values.job_count[1] = 0;
        wavefront_state_values.index[0] = cast(u32*, state_index[0]);
        wavefront_state_values.index[1] = cast(u32*, state_index[1]);
        wavefront_state_values.states = cast(wavefront::PathState*, pathstate_buffer);

        CUDA_CHECK(cuMemcpyHtoD(wavefront_state, &wavefront_state_values, sizeof(wavefront::State)));
        CUDA_CHECK(cuMemsetD8(frame_buffer, 0, settings.buffer_width * settings.buffer_height * sizeof(Vector4)));

        const u32 block_x = 64;
        const u32 grid_x = (settings.buffer_width * settings.buffer_height + block_x - 1) / block_x;

        const u64 wavefront_start_time = SDL_GetPerformanceCounter();

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

            CUDA_CHECK(cuLaunchKernel(reset,
                           1, 1, 1,
                           1, 1, 1,
                           0,
                           0,
                           reset_params,
                           NULL));
             
#if 0
            {
                const u32 block_x = 32;
                const u32 grid_x = 512;
                                                                         
                void* extend_rays_params[] = {
                    &wavefront_state,
                    &current,
                    &cbvh.data,
                    &gpu_cbvh,
                    &gpu_triangles,
                };
                CUDA_CHECK(cuLaunchKernel(extend_rays_compressed_fetch, 
                                          grid_x, 1, 1, 
                                          block_x, 1, 1, 
                                          0, 
                                          0, 
                                          extend_rays_params, 
                                          NULL));
            }
#endif

#if 1
            void* extend_rays_params[] = {
                &wavefront_state, 
                &current, 
                &cbvh.data,
                &gpu_cbvh, 
                &gpu_triangles,
            };

            CUDA_CHECK(cuLaunchKernel(extend_rays_compressed, 
                                      grid_x, 1, 1, 
                                      block_x, 1, 1, 
                                      0, 
                                      0, 
                                      extend_rays_params, 
                                      NULL));
#endif

            void* shade_params[] = {
                &wavefront_state, 
                &current, 
                &gpu_triangles, 
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

        void* output_to_buffer_params[] = {
            &wavefront_state,
            &frame_buffer,
            &settings.buffer_width,
            &settings.buffer_height,
        };

        CUDA_CHECK(cuLaunchKernel(output_to_buffer,
                       grid_x, 1, 1,
                       block_x, 1, 1,
                       0,
                       0,
                       output_to_buffer_params,
                       NULL));

        void* reset_params[] = {
            &wavefront_state,
            &current,
        };

        CUDA_CHECK(cuLaunchKernel(reset,
                       1, 1, 1,
                       1, 1, 1,
                       0,
                       0,
                       reset_params,
                       NULL));

        CUDA_CHECK(cuMemcpyDtoH(&wavefront_state_values, wavefront_state, sizeof(wavefront::State)));

        const u64 wavefront_end_time = SDL_GetPerformanceCounter();

        const f32 wavefront_frame_time = cast(f32, wavefront_end_time - wavefront_start_time) / frequency;
        const f32 megarays_this_frame = cast(f32, wavefront_state_values.total_ray_count) / 1000000.0f;
        const f32 megarays_per_second = megarays_this_frame / wavefront_frame_time;

        {
            const u32 block_x = 16, block_y = 16;
            const u32 grid_x = (settings.buffer_width + block_x - 1) / block_x; 
            const u32 grid_y = (settings.buffer_height + block_y - 1) / block_y;
            void* accumulate_params[] = {
                &frame_buffer, &accumulator, &screen_buffer, &settings.buffer_width, &settings.buffer_height, &acc_frame,
            };
            CUDA_CHECK(cuLaunchKernel(accumulate, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, accumulate_params, NULL));
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
            ImGui::Text("%7.3f MRays/s", megarays_per_second);
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

            DragFloatPrecise("Camera speed", &camera_speed);

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

