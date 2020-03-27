#pragma once

namespace wavefront {
    struct alignas(16) PathState {
        vec3 ray_pos; u32 pixel_index; // 16 bytes
        vec3 ray_dir; RngXor32 rng; // 16 bytes

        vec3 throughput; // 12 bytes
        u32 pad0; // 4 bytes

        f32 t, u, v; u32 triangle_id; // 16 bytes

#ifdef __CUDACC__
        __device__ bool hit() const { return triangle_id != UINT32_MAX; }
#endif
    }; // total 64 bytes

    struct alignas(16) State {
        u32 job_count[2];

        PathState* states[2];
    };
}
