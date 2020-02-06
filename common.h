#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

typedef uint32_t u32;

#define cast(Type, Expr) ((Type)(Expr))

#define UNIMPLEMENTED() 

#define UNUSED(x) ((void)(x))

#endif // COMMON_H
