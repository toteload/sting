#include <math.h>

__global__ void cuda_add(float* a, float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) { *c = *a + *b; }
}

extern "C" void add_arrays(float* a, float* b, float* c, int n) {
    float* da, *db, *dc;

    cudaMalloc(&da, n*sizeof(float));
    cudaMalloc(&db, n*sizeof(float));
    cudaMalloc(&dc, n*sizeof(float));

    cudaMemcpy(da, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid_size = (int)ceil((float)n/block_size);

    cuda_add<<<grid_size, block_size>>>(da, db, dc, n);

    cudaMemcpy(c, dc, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
