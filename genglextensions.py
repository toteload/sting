# --------------------------------------------------------------------------- #
# Made by David Bos :)
# --------------------------------------------------------------------------- #

EXTENSIONS_FILE_NAME = "glextensions.txt"
GETPROCADDRESS_FUNCTION = "SDL_GL_GetProcAddress"

# --------------------------------------------------------------------------- #

out_fmt_h = """#pragma once

// This header file was automatically generated

{}

bool load_gl_extensions();
"""

# --------------------------------------------------------------------------- #

out_fmt_cpp = """
// This source file was automatically generated

{}

bool load_gl_extensions() {{
    // Load all the functions
{}

    // Check if all the functions were correctly loaded
{}

    return true;
}}
"""

functions_to_load = [e.rstrip() for e in open(EXTENSIONS_FILE_NAME, "r").readlines() if e.strip()]
glext = open("glext.h", "r").readlines()

# Sort the functions to load alphabetically because it looks neato
functions_to_load.sort()

# Find all the functions declared in glext.h
glext_functions = { }
for line in glext:
    parts = line.split(" ")
    if parts[0] == "GLAPI" and "APIENTRY" in parts:
        for p in parts:
            if p.startswith("gl"):
                glext_functions[p] = True

# Make sure that the requested functions can be found in glext.h
for fn in functions_to_load:
    if fn not in glext_functions:
        print("{} not found in glext.h, maybe this was a typo?".format(fn))

def load_ext_string(ext):
    exttype = "PFN" + ext.upper() + "PROC"
    return "    {ext} = ({exttype}){getprocaddress}(\"{ext}\");".format(ext=ext, 
                                                                        getprocaddress=GETPROCADDRESS_FUNCTION, 
                                                                        exttype=exttype)

def check_ext_string(ext):
    return "    if (!{ext}) {{ return false; }}".format(ext=ext)

function_definitions = ["PFN" + e.upper() + "PROC" + " " + e + ";" for e in functions_to_load]

open("gl_extension_loader.cpp", "w").write(out_fmt_cpp.format("\n".join(function_definitions), 
                                                              "\n".join(list(map(load_ext_string,  functions_to_load))),
                                                              "\n".join(list(map(check_ext_string, functions_to_load)))))

declarations = "\n".join(["extern " + fn for fn in function_definitions])
open("gl_extension_loader.h", "w").write(out_fmt_h.format(declarations))
