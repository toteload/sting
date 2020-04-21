#pragma once

#include "dab.h"

#define dab_reflect

#define dab_reflect_member_count(StructType) dab_reflect__member_count_##StructType
#define dab_reflect_member(StructType, Index) dab_reflect__memberinfo_##StructType[Index]
 
namespace dab { namespace reflect {

struct TypeInfo {
    enum Type : u32 {
        Char,
        I32,
        U32,
        F32,
        Vector2,
        Vector3,
        Pointer,
        Array,

        Undefined,
    };

    Type type;
    union {
        struct { Type pointer_type; };
        struct { u32 array_size; Type array_type; };
    };
};

struct StructMemberInfo {
    const char* name;
    TypeInfo type_info;
    u32 offset;
};                                                                    

} }
