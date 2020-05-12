import re
import datetime

# --------------------------------------------------------------------------- #
# Made by David Bos :)
# --------------------------------------------------------------------------- #

# Options
# --------------------------------------------------------------------------- #

SOURCE_FILES_TO_SCAN = [ "src/main.cpp", "Z:/dab/dab.h", "ext/imgui/imgui_impl_opengl3.cpp" ]
GETPROCADDRESS_FUNCTION = "SDL_GL_GetProcAddress"
VERBOSE = False

USE_EXTENSIONS_FILE_NAME = False
EXTENSIONS_FILE_NAME = "glextensions.txt"

# --------------------------------------------------------------------------- #

default_gl_functions = [ 'glColor3usv', 'glMultMatrixf', 'glEnd', 'glPolygonOffset', 'glTexCoord1dv', 
        'glPolygonStipple', 'glMap1f', 'glEnableClientState', 'glGetTexImage', 'glClearAccum', 'glRasterPos4f', 
        'glMaterialfv', 'glIndexiv', 'glColor3dv', 'glTranslated', 'glColor4bv', 'glIndexd', 'glTexCoord2dv', 
        'glVertex2d', 'glTexCoord3sv', 'glTexEnviv', 'glTexCoord1d', 'glColor3uiv', 'glLoadMatrixf', 
        'glTexParameteriv', 'glColor4fv', 'glMapGrid2d', 'glEvalCoord1f', 'glPointSize', 'glReadPixels', 
        'glColor3s', 'glTexCoord1fv', 'glTexParameteri', 'glColor3ubv', 'glNormal3bv', 'glVertex4f', 'glVertex4dv', 
        'glDrawArrays', 'glFeedbackBuffer', 'glLightModeli', 'glEvalMesh1', 'glPixelMapusv', 'glCopyPixels', 
        'glVertex3dv', 'glMap2f', 'glAreTexturesResident', 'glRasterPos4i', 'glMap2d', 'glNormalPointer', 
        'glBlendFunc', 'glVertex3sv', 'glColor4usv', 'glIndexPointer', 'glBindTexture', 'glRasterPos4s', 
        'glRasterPos3sv', 'glGenLists', 'glDeleteLists', 'glGetError', 'glLineWidth', 'glInitNames', 
        'glColor4b', 'glTexGend', 'glTexCoord3dv', 'glVertex4iv', 'glIndexsv', 'glVertex3fv', 
        'glRasterPos3iv', 'glOrtho', 'glPixelTransferf', 'glGetString', 'glGetTexGenfv', 'glClearColor', 
        'glGetClipPlane', 'glRasterPos4fv', 'glRasterPos2sv', 'glPassThrough', 'glRasterPos3dv', 'glRecti', 
        'glVertex4fv', 'glNormal3i', 'glScissor', 'glAlphaFunc', 'glLoadIdentity', 'glPushMatrix', 
        'glDepthRange', 'glTexCoord2s', 'glRasterPos3f', 'glColor3iv', 'glGetMapfv', 'glColor3b', 
        'glViewport', 'glGetDoublev', 'glGetPixelMapuiv', 'glReadBuffer', 'glTexCoord4sv', 'glColor4s', 
        'glTexCoord4fv', 'glTexCoord3fv', 'glDeleteTextures', 'glRasterPos2d', 'glCopyTexImage1D', 'glDisable', 
        'glTexGeni', 'glDisableClientState', 'glGetFloatv', 'glLightfv', 'glClearStencil', 'glGetTexParameterfv', 
        'glMapGrid1f', 'glRectfv', 'glTexCoord2sv', 'glCopyTexSubImage1D', 'glNormal3s', 'glColor4iv', 
        'glTexCoord2i', 'glNormal3iv', 'glTexCoord4i', 'glVertex2dv', 'glVertex2iv', 'glLoadName', 'glTexEnvi', 
        'glHint', 'glScalef', 'glRasterPos4dv', 'glEvalCoord1fv', 'glVertex2s', 'glIndexubv', 'glTexCoord2iv', 
        'glEvalCoord2fv', 'glEvalCoord2dv', 'glRotated', 'glColor4us', 'glColor3sv', 'glRotatef', 
        'glTexCoordPointer', 'glPixelMapfv', 'glTexCoord2f', 'glFogiv', 'glEndList', 'glColor3i', 
        'glTexImage2D', 'glVertex3i', 'glEvalCoord1d', 'glTexImage1D', 'glVertexPointer', 'glClipPlane', 
        'glGetMaterialfv', 'glVertex3iv', 'glTexCoord4iv', 'glRasterPos4iv', 'glRasterPos3i', 'glLightModeliv', 
        'glVertex4sv', 'glMateriali', 'glPixelTransferi', 'glCopyTexImage2D', 'glRasterPos3fv', 'glVertex4i', 
        'glArrayElement', 'glLightModelfv', 'glLineStipple', 'glDepthFunc', 'glRasterPos2iv', 'glVertex3d', 
        'glColor4ub', 'glTexCoord4s', 'glAccum', 'glTexCoord4f', 'glRasterPos2i', 'glNormal3b', 'glRasterPos3s', 
        'glTexCoord4d', 'glPopMatrix', 'glMaterialf', 'glIsList', 'glGetPixelMapusv', 'glMap1d', 'glGetTexEnvfv', 
        'glIndexdv', 'glGetPolygonStipple', 'glGetLightfv', 'glRectdv', 'glGetTexLevelParameteriv', 'glStencilMask', 
        'glRectf', 'glNormal3d', 'glEdgeFlag', 'glEvalCoord1dv', 'glPrioritizeTextures', 'glColorMaterial', 
        'glPopName', 'glClearDepth', 'glGetLightiv', 'glRectd', 'glNormal3dv', 'glEvalMesh2', 'glColor4ubv', 
        'glBitmap', 'glGenTextures', 'glTexCoord1f', 'glLoadMatrixd', 'glTexEnvf', 'glPopAttrib', 
        'glGetMaterialiv', 'glEdgeFlagv', 'glLightModelf', 'glFogfv', 'glVertex3f', 'glGetIntegerv', 
        'glDrawPixels', 'glGetMapdv', 'glLightf', 'glBegin', 'glGetTexEnviv', 'glNewList', 'glNormal3fv', 
        'glVertex4d', 'glCullFace', 'glColor4d', 'glIndexub', 'glFogi', 'glVertex2fv', 'glMaterialiv', 
        'glTexCoord3iv', 'glTexParameterf', 'glRenderMode', 'glFrustum', 'glPushName', 'glPushClientAttrib', 
        'glGetPointerv', 'glLogicOp', 'glTexGenfv', 'glColor3ui', 'glRasterPos4sv', 'glShadeModel', 
        'glEvalCoord2f', 'glTexSubImage1D', 'glPixelZoom', 'glColor3us', 'glEnable', 'glIndexs', 'glColor3f', 
        'glTexCoord4dv', 'glTexCoord2fv', 'glNormal3f', 'glTexGenf', 'glRasterPos2s', 'glColor4ui', 'glCallList', 
        'glTexEnvfv', 'glDepthMask', 'glRasterPos4d', 'glNormal3sv', 'glCopyTexSubImage2D', 'glFrontFace', 
        'glRasterPos2fv', 'glColor3ub', 'glStencilFunc', 'glRasterPos2dv', 'glGetTexGendv', 'glPopClientAttrib', 
        'glTexCoord1iv', 'glVertex2f', 'glColor4i', 'glColor4uiv', 'glDrawBuffer', 'glGetTexGeniv', 'glVertex2sv', 
        'glColorPointer', 'glVertex4s', 'glEvalCoord2d', 'glTexSubImage2D', 'glVertex3s', 'glGetMapiv', 
        'glGetTexParameteriv', 'glClearIndex', 'glVertex2i', 'glRasterPos3d', 'glTexCoord3f', 'glColor3fv', 
        'glGetBooleanv', 'glColor3bv', 'glTexCoord1i', 'glPushAttrib', 'glFogf', 'glFlush', 'glRasterPos2f', 
        'glColor4sv', 'glPixelStoref', 'glPixelStorei', 'glTexCoord3s', 'glLighti', 'glTexCoord2d', 
        'glTexCoord3d', 'glMapGrid1d', 'glTexCoord1sv', 'glTexGeniv', 'glIndexfv', 'glIndexf', 'glEvalPoint2', 
        'glStencilOp', 'glInterleavedArrays', 'glIndexi', 'glRects', 'glScaled', 'glGetPixelMapfv', 
        'glIsEnabled', 'glIsTexture', 'glColor3d', 'glMatrixMode', 'glTexCoord3i', 'glClear', 
        'glEdgeFlagPointer', 'glTranslatef', 'glGetTexLevelParameterfv', 'glColor4f', 'glTexCoord1s', 
        'glDrawElements', 'glLightiv', 'glSelectBuffer', 'glPixelMapuiv', 'glPolygonMode', 'glTexGendv', 'glColor4dv', 
        'glTexParameterfv', 'glMapGrid2f', 'glFinish', 'glRectsv', 'glEvalPoint1', 'glColorMask', 'glRectiv', 
        'glCallLists', 'glListBase', 'glMultMatrixd', 'glIndexMask']

# --------------------------------------------------------------------------- #

script_msg = """// This file has been generate by porky.py on {}""".format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M"))

out_fmt_h = """#pragma once

{}

{}

int porky_load_extensions();
"""

# --------------------------------------------------------------------------- #

out_fmt_cpp = """{}

{}

int porky_load_extensions() {{
{}

{}

    return 1;
}}
"""

functions_to_load = None

if USE_EXTENSIONS_FILE_NAME:
    functions_to_load = [e.rstrip() for e in open(EXTENSIONS_FILE_NAME, "r").readlines() if e.strip()]
else:
    gl_functions = set()
    for f in SOURCE_FILES_TO_SCAN:
        for line in open(f, "r").readlines():
            gl_functions.update(re.findall('gl[A-Z][A-Za-z0-9]*', line))
    functions_to_load = list(gl_functions)

# Sort the functions to load alphabetically because it looks neato
functions_to_load.sort()

if VERBOSE:
    print("OpenGL functions found:")
    for fn in functions_to_load:
        print(fn)

glext = open("ext/glext.h", "r").readlines()

# Find all the functions declared in glext.h or GL.h
glext_functions = { }
for line in glext:
    parts = line.split(" ")
    if parts[0] == "GLAPI" and "APIENTRY" in parts:
        for p in parts:
            if p.startswith("gl"):
                glext_functions[p] = True

# Remove all the OpenGL 1 functions 
functions_to_load = list(filter(lambda x: x not in default_gl_functions, functions_to_load))

# Make sure that the requested functions can be found in glext.h
# We do this to let the user know that a function could not be found
for fn in functions_to_load:
    if fn not in glext_functions:
        print("{} not found in glext.h or GL.h, maybe this was a typo? It could also be a false positive from this script...".format(fn))

functions_to_load = list(filter(lambda x: x in glext_functions, functions_to_load))

def load_ext_string(ext):
    exttype = "PFN" + ext.upper() + "PROC"
    return "    {ext} = ({exttype}){getprocaddress}(\"{ext}\");".format(ext=ext, 
                                                                        getprocaddress=GETPROCADDRESS_FUNCTION, 
                                                                        exttype=exttype)

def check_ext_string(ext):
    return "    if (!{ext}) {{ return 0; }}".format(ext=ext)

function_definitions = ["PFN" + e.upper() + "PROC" + " " + e + ";" for e in functions_to_load]

open("porky_load.cpp", "w").write(out_fmt_cpp.format(script_msg,
                                                     "\n".join(function_definitions), 
                                                     "\n".join(list(map(load_ext_string,  functions_to_load))),
                                                     "\n".join(list(map(check_ext_string, functions_to_load)))))

declarations = "\n".join(["extern " + fn for fn in function_definitions])
open("porky_load.h", "w").write(out_fmt_h.format(script_msg, declarations))
