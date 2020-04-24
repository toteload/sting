#pragma once

namespace wavefront {
    struct alignas(16) PathState {
        Vector3 ray_pos; u32 pixel_index; // 16 bytes
        Vector3 ray_dir; RngXor32 rng; // 16 bytes

        Vector3 throughput; u32 pad0; // 16 bytes
        Vector3 emission;   u32 pad1; // 16 bytes

        f32 t, u, v; u32 triangle_id; // 16 bytes

#ifdef __CUDACC__
        __device__ bool hit() const { return triangle_id != UINT32_MAX; }
#endif
    }; // total 80 bytes

    struct alignas(16) State {
        u32 total_ray_count;
        u32 job_count[2];

        PathState* states[2];
    };

    struct alignas(16) State_v2 {
        u32 job_count[2];
        u32* state_index[2];
        PathState* states;
    };
}
