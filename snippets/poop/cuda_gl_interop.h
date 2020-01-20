#ifndef CUDA_GL_INTEROP_H
#define CUDA_GL_INTEROP_H

typedef struct cuda_gl_interop cuda_gl_interop;

struct cuda_gl_interop {
    u32 width;
    u32 height;

    GLuint fb;
    GLuint rb;

    cudaGraphicsResource* g;
    cudaArray* a;
};

void cuda_gl_interop_create(cuda_gl_interop* interop, u32 width, u32 height);
void cuda_gl_interop_destroy(cuda_gl_interop* interop);
void cuda_gl_interop_blit(cuda_gl_interop* interop);

#endif // CUDA_GL_INTEROP_H
