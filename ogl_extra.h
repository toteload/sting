#ifndef GUARD_OGL_EXTRA_H_INCLUDED
#define GUARD_OGL_EXTRA_H_INCLUDED

#define APIENTRYP   APIENTRY *

// Typedefs
typedef GLuint (APIENTRYP PFNGLCREATESHADERPROC) (GLenum type);
typedef void (APIENTRYP PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
typedef void (APIENTRYP PFNGLCOMPILESHADERPROC) (GLuint shader);
typedef void (APIENTRYP PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRYP PFNGLDELETESHADERPROC) (GLuint shader);
typedef GLuint (APIENTRYP PFNGLCREATEPROGRAMPROC) (void);
typedef void (APIENTRYP PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (APIENTRYP PFNGLLINKPROGRAMPROC) (GLuint program);
typedef void (APIENTRYP PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRYP PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *arrays);
typedef void (APIENTRYP PFNGLBINDVERTEXARRAYPROC) (GLuint array);
typedef void (APIENTRYP PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (APIENTRYP PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRYP PFNGLBUFFERDATAPROC) (GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRYP PFNGLVERTEXATTRIBPOINTERPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
typedef void (APIENTRYP PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (APIENTRYP PFNGLUSEPROGRAMPROC) (GLuint program);
typedef void (APIENTRYP PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (APIENTRYP PFNGLBINDATTRIBLOCATIONPROC) (GLuint program, GLuint index, const GLchar *name);
typedef void (APIENTRYP PFNGLUNIFORM1UIPROC) (GLint location, GLuint v0);
typedef void (APIENTRYP PFNGLUNIFORM3FVPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (APIENTRYP PFNGLUNIFORM4FVPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (APIENTRYP PFNGLUNIFORM2UIPROC) (GLint location, GLuint v0, GLuint v1);
typedef void (APIENTRYP PFNGLUNIFORMMATRIX4FVPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef GLint (APIENTRYP PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const GLchar *name);
typedef void (APIENTRYP PFNGLDELETEVERTEXARRAYSPROC) (GLsizei n, const GLuint *arrays);
typedef void (APIENTRYP PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);
typedef void (APIENTRYP PFNGLCREATERENDERBUFFERSPROC) (GLsizei n, GLuint *renderbuffers);
typedef void (APIENTRYP PFNGLCREATEFRAMEBUFFERSPROC) (GLsizei n, GLuint *framebuffers);
typedef void (APIENTRYP PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC) (GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
typedef void (APIENTRYP PFNGLNAMEDRENDERBUFFERSTORAGEPROC) (GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height);
typedef void (APIENTRYP PFNGLBLITNAMEDFRAMEBUFFERPROC) (GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);

// Function declarations
PFNGLCREATESHADERPROC glCreateShader;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLDELETESHADERPROC glDeleteShader;
PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLGETPROGRAMIVPROC glGetProgramiv;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLBINDBUFFERPROC glBindBuffer;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocation;
PFNGLUNIFORM1UIPROC glUniform1ui;
PFNGLUNIFORM3FVPROC glUniform3fv;
PFNGLUNIFORM4FVPROC glUniform4fv;
PFNGLUNIFORM2UIPROC glUniform2ui;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
PFNGLDELETEBUFFERSPROC glDeleteBuffers;
PFNGLCREATERENDERBUFFERSPROC glCreateRenderbuffers;
PFNGLCREATEFRAMEBUFFERSPROC glCreateFramebuffers;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC glNamedFramebufferRenderbuffer;
PFNGLNAMEDRENDERBUFFERSTORAGEPROC glNamedRenderbufferStorage;
PFNGLBLITNAMEDFRAMEBUFFERPROC glBlitNamedFramebuffer;

u32 ogl_extra_init() {
    glCreateShader = (PFNGLCREATESHADERPROC) wglGetProcAddress("glCreateShader");
    if (!glCreateShader) { return false; }

    glShaderSource = (PFNGLSHADERSOURCEPROC) wglGetProcAddress("glShaderSource");
    if (!glShaderSource) { return false; }

    glCompileShader = (PFNGLCOMPILESHADERPROC) wglGetProcAddress("glCompileShader");
    if (!glCompileShader) { return false; }

    glGetShaderiv = (PFNGLGETSHADERIVPROC) wglGetProcAddress("glGetShaderiv");
    if (!glGetShaderiv) { return false; }

    glDeleteShader = (PFNGLDELETESHADERPROC) wglGetProcAddress("glDeleteShader");
    if (!glDeleteShader) { return false; }

    glCreateProgram = (PFNGLCREATEPROGRAMPROC) wglGetProcAddress("glCreateProgram");
    if (!glCreateProgram) { return false; }

    glAttachShader = (PFNGLATTACHSHADERPROC) wglGetProcAddress("glAttachShader");
    if (!glAttachShader) { return false; }

    glLinkProgram = (PFNGLLINKPROGRAMPROC) wglGetProcAddress("glLinkProgram");
    if (!glLinkProgram) { return false; }

    glGetProgramiv = (PFNGLGETPROGRAMIVPROC) wglGetProcAddress("glGetProgramiv");
    if (!glGetProgramiv) { return false; }

    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC) wglGetProcAddress("glGenVertexArrays");
    if (!glGenVertexArrays) { return false; }

    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC) wglGetProcAddress("glBindVertexArray");
    if (!glBindVertexArray) { return false; }

    glGenBuffers = (PFNGLGENBUFFERSPROC) wglGetProcAddress("glGenBuffers");
    if (!glGenBuffers) { return false; }

    glBindBuffer = (PFNGLBINDBUFFERPROC) wglGetProcAddress("glBindBuffer");
    if (!glBindBuffer) { return false; }

    glBufferData = (PFNGLBUFFERDATAPROC) wglGetProcAddress("glBufferData");
    if (!glBufferData) { return false; }

    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC) wglGetProcAddress("glVertexAttribPointer");
    if (!glVertexAttribPointer) { return false; }

    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC) wglGetProcAddress("glEnableVertexAttribArray");
    if (!glEnableVertexAttribArray) { return false; }

    glUseProgram = (PFNGLUSEPROGRAMPROC) wglGetProcAddress("glUseProgram");
    if (!glUseProgram) { return false; }

    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC) wglGetProcAddress("glGetShaderInfoLog");
    if (!glGetShaderInfoLog) { return false; }

    glBindAttribLocation = (PFNGLBINDATTRIBLOCATIONPROC) wglGetProcAddress("glBindAttribLocation");
    if (!glBindAttribLocation) { return false; }

    glUniform1ui = (PFNGLUNIFORM1UIPROC) wglGetProcAddress("glUniform1ui");
    if (!glUniform1ui) { return false; }

    glUniform3fv = (PFNGLUNIFORM3FVPROC) wglGetProcAddress("glUniform3fv");
    if (!glUniform3fv) { return false; }

    glUniform4fv = (PFNGLUNIFORM4FVPROC) wglGetProcAddress("glUniform4fv");
    if (!glUniform4fv) { return false; }

    glUniform2ui = (PFNGLUNIFORM2UIPROC) wglGetProcAddress("glUniform2ui");
    if (!glUniform2ui) { return false; }

    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC) wglGetProcAddress("glUniformMatrix4fv");
    if (!glUniformMatrix4fv) { return false; }

    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC) wglGetProcAddress("glGetUniformLocation");
    if (!glGetUniformLocation) { return false; }

    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC) wglGetProcAddress("glDeleteVertexArrays");
    if (!glDeleteVertexArrays) { return false; }

    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC) wglGetProcAddress("glDeleteBuffers");
    if (!glDeleteBuffers) { return false; }

    glCreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC) wglGetProcAddress("glCreateRenderbuffers");
    if (!glCreateRenderbuffers) { return false; }

    glCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC) wglGetProcAddress("glCreateFramebuffers");
    if (!glCreateFramebuffers) { return false; }

    glNamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC) wglGetProcAddress("glNamedFramebufferRenderbuffer");
    if (!glNamedFramebufferRenderbuffer) { return false; }

    glNamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC) wglGetProcAddress("glNamedRenderbufferStorage");
    if (!glNamedRenderbufferStorage) { return false; }

    glBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC) wglGetProcAddress("glBlitNamedFramebuffer");
    if (!glBlitNamedFramebuffer) { return false; }

    return 1;
}

#endif // GUARD_OGL_EXTRA_H_INCLUDED