extensions   = [l.rstrip() for l in open("gl_extensions.txt", "r").readlines() if l.strip()]
ext_types    = ["PFN" + ext.upper() + "PROC" for ext in extensions]
declarations = [pair[0] + " " + pair[1] + ";" for pair in zip(ext_types, extensions)]
typedefs     = []

glext  = open("glext.h", "r").readlines()

# --------------------------------------------------------------------------- #
# Find all the typedefs for the extension functions in glext.h                #
# --------------------------------------------------------------------------- #
for t in ext_types:
    found = False
    for line in glext:
        index = line.find(t)
        if index != -1 and line.startswith("typedef"):
            typedefs.append(line.rstrip())
            found = True
            break
    if not found:
        print("Could not find type {}!".format(t))

# --------------------------------------------------------------------------- #
#                           Header file generation                            #
# --------------------------------------------------------------------------- #
header = """\
#ifndef GUARD_OGL_EXTRA_H_INCLUDED
#define GUARD_OGL_EXTRA_H_INCLUDED

#define APIENTRYP   APIENTRY *

// Typedefs
{}

// Function declarations
{}

u32 ogl_extra_init() {{
{}
    return 1;
}}

#endif // GUARD_OGL_EXTRA_H_INCLUDED"""

proc_load = """\
    {ext} = ({ext_type}) wglGetProcAddress("{ext}");
    if (!{ext}) {{ return false; }}
"""

pairs = list(zip(ext_types, extensions))
loads = [proc_load.format(ext=pair[1], ext_type=pair[0]) for pair in pairs]

header_file = open("ogl_extra.h", "w")
header_file.write(header.format("\n".join(typedefs), 
                                "\n".join(declarations),
                                "\n".join(loads)))

print("ogl_extra.h generated for {} functions!".format(len(pairs)))

