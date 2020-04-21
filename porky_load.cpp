// This file has been generate by porky.py on 21/04/2020 14:44

PFNGLACTIVETEXTUREPROC glActiveTexture;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLBINDBUFFERPROC glBindBuffer;
PFNGLBINDSAMPLERPROC glBindSampler;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
PFNGLBLENDEQUATIONPROC glBlendEquation;
PFNGLBLENDEQUATIONSEPARATEPROC glBlendEquationSeparate;
PFNGLBLENDFUNCSEPARATEPROC glBlendFuncSeparate;
PFNGLBLITNAMEDFRAMEBUFFERPROC glBlitNamedFramebuffer;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLCLIPCONTROLPROC glClipControl;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLCREATEFRAMEBUFFERSPROC glCreateFramebuffers;
PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLCREATERENDERBUFFERSPROC glCreateRenderbuffers;
PFNGLCREATESHADERPROC glCreateShader;
PFNGLDELETEBUFFERSPROC glDeleteBuffers;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
PFNGLDELETEPROGRAMPROC glDeleteProgram;
PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
PFNGLDELETESHADERPROC glDeleteShader;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
PFNGLDETACHSHADERPROC glDetachShader;
PFNGLDRAWELEMENTSBASEVERTEXPROC glDrawElementsBaseVertex;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
PFNGLGETATTRIBLOCATIONPROC glGetAttribLocation;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
PFNGLGETPROGRAMIVPROC glGetProgramiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC glNamedFramebufferRenderbuffer;
PFNGLNAMEDRENDERBUFFERSTORAGEPROC glNamedRenderbufferStorage;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLUNIFORM1IPROC glUniform1i;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;

int porky_load_extensions() {
    glActiveTexture = (PFNGLACTIVETEXTUREPROC)SDL_GL_GetProcAddress("glActiveTexture");
    glAttachShader = (PFNGLATTACHSHADERPROC)SDL_GL_GetProcAddress("glAttachShader");
    glBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    glBindSampler = (PFNGLBINDSAMPLERPROC)SDL_GL_GetProcAddress("glBindSampler");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)SDL_GL_GetProcAddress("glBindVertexArray");
    glBlendEquation = (PFNGLBLENDEQUATIONPROC)SDL_GL_GetProcAddress("glBlendEquation");
    glBlendEquationSeparate = (PFNGLBLENDEQUATIONSEPARATEPROC)SDL_GL_GetProcAddress("glBlendEquationSeparate");
    glBlendFuncSeparate = (PFNGLBLENDFUNCSEPARATEPROC)SDL_GL_GetProcAddress("glBlendFuncSeparate");
    glBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC)SDL_GL_GetProcAddress("glBlitNamedFramebuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    glClipControl = (PFNGLCLIPCONTROLPROC)SDL_GL_GetProcAddress("glClipControl");
    glCompileShader = (PFNGLCOMPILESHADERPROC)SDL_GL_GetProcAddress("glCompileShader");
    glCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glCreateFramebuffers");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)SDL_GL_GetProcAddress("glCreateProgram");
    glCreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glCreateRenderbuffers");
    glCreateShader = (PFNGLCREATESHADERPROC)SDL_GL_GetProcAddress("glCreateShader");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");
    glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteFramebuffers");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)SDL_GL_GetProcAddress("glDeleteProgram");
    glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteRenderbuffers");
    glDeleteShader = (PFNGLDELETESHADERPROC)SDL_GL_GetProcAddress("glDeleteShader");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glDeleteVertexArrays");
    glDetachShader = (PFNGLDETACHSHADERPROC)SDL_GL_GetProcAddress("glDetachShader");
    glDrawElementsBaseVertex = (PFNGLDRAWELEMENTSBASEVERTEXPROC)SDL_GL_GetProcAddress("glDrawElementsBaseVertex");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    glGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)SDL_GL_GetProcAddress("glGenVertexArrays");
    glGetAttribLocation = (PFNGLGETATTRIBLOCATIONPROC)SDL_GL_GetProcAddress("glGetAttribLocation");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)SDL_GL_GetProcAddress("glGetProgramInfoLog");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)SDL_GL_GetProcAddress("glGetProgramiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)SDL_GL_GetProcAddress("glGetShaderiv");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)SDL_GL_GetProcAddress("glGetUniformLocation");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)SDL_GL_GetProcAddress("glLinkProgram");
    glNamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC)SDL_GL_GetProcAddress("glNamedFramebufferRenderbuffer");
    glNamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC)SDL_GL_GetProcAddress("glNamedRenderbufferStorage");
    glShaderSource = (PFNGLSHADERSOURCEPROC)SDL_GL_GetProcAddress("glShaderSource");
    glUniform1i = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)SDL_GL_GetProcAddress("glUniformMatrix4fv");
    glUseProgram = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribPointer");

    if (!glActiveTexture) { return 0; }
    if (!glAttachShader) { return 0; }
    if (!glBindBuffer) { return 0; }
    if (!glBindSampler) { return 0; }
    if (!glBindVertexArray) { return 0; }
    if (!glBlendEquation) { return 0; }
    if (!glBlendEquationSeparate) { return 0; }
    if (!glBlendFuncSeparate) { return 0; }
    if (!glBlitNamedFramebuffer) { return 0; }
    if (!glBufferData) { return 0; }
    if (!glClipControl) { return 0; }
    if (!glCompileShader) { return 0; }
    if (!glCreateFramebuffers) { return 0; }
    if (!glCreateProgram) { return 0; }
    if (!glCreateRenderbuffers) { return 0; }
    if (!glCreateShader) { return 0; }
    if (!glDeleteBuffers) { return 0; }
    if (!glDeleteFramebuffers) { return 0; }
    if (!glDeleteProgram) { return 0; }
    if (!glDeleteRenderbuffers) { return 0; }
    if (!glDeleteShader) { return 0; }
    if (!glDeleteVertexArrays) { return 0; }
    if (!glDetachShader) { return 0; }
    if (!glDrawElementsBaseVertex) { return 0; }
    if (!glEnableVertexAttribArray) { return 0; }
    if (!glGenBuffers) { return 0; }
    if (!glGenVertexArrays) { return 0; }
    if (!glGetAttribLocation) { return 0; }
    if (!glGetProgramInfoLog) { return 0; }
    if (!glGetProgramiv) { return 0; }
    if (!glGetShaderInfoLog) { return 0; }
    if (!glGetShaderiv) { return 0; }
    if (!glGetUniformLocation) { return 0; }
    if (!glLinkProgram) { return 0; }
    if (!glNamedFramebufferRenderbuffer) { return 0; }
    if (!glNamedRenderbufferStorage) { return 0; }
    if (!glShaderSource) { return 0; }
    if (!glUniform1i) { return 0; }
    if (!glUniformMatrix4fv) { return 0; }
    if (!glUseProgram) { return 0; }
    if (!glVertexAttribPointer) { return 0; }

    return 1;
}
