#pragma once

namespace wavefront {
    struct alignas(16) PathState {
        Vector3 ray_pos; RngXor32 rng; // 16 bytes
        Vector3 ray_dir; f32      t;   // 16 bytes

        Vector3 throughput; f32 u; // 16 bytes
        Vector3 emission;   f32 v; // 16 bytes

        u32 triangle_id; u32 pad0, pad1, pad2; // 16 bytes

#ifdef __CUDACC__
        __device__ bool hit() const { return triangle_id != UINT32_MAX; }
#endif
    }; // total 80 bytes

    struct alignas(16) State {
        u32 total_ray_count;

        u32 jobi;
        u32 job_count[2];
        u32* index[2];
        PathState* states;
    };
}
