out_fmt = """{}

uint32_t load_gl_extensions() {{
{}
    return 1;
}}
"""

def make_fn_check(ext):
    exttype = "PFN" + ext.upper() + "PROC"
    return "    {ext} = ({exttype})SDL_GL_GetProcAddress(\"{ext}\");\n    if (!{ext}) {{ return 0; }}\n".format(ext=ext, 
                                                                                                               exttype=exttype)

extensions = [e.rstrip() for e in open("glextensions.txt", "r").readlines() if e.strip()]
declarations = "\n".join(["PFN" + e.upper() + "PROC" + " " + e + ";" for e in extensions])
open("load_gl_extensions.h", "w").write(out_fmt.format(declarations, "\n".join(list(map(make_fn_check, extensions)))))
