#include "dab.h"
#include "dab_reflect.h"

#include "cpp_lexer.h"
#include "cpp_lexer.cpp"

#include <vector>
#include <stdio.h>
  
using namespace dab::reflect;

FILE* out;
 
const char* type_to_string(TypeInfo::Type t) {
    switch (t) {
    case TypeInfo::Char: return "Char";
    case TypeInfo::I32: return "I32";
    case TypeInfo::U32: return "U32";
    case TypeInfo::F32: return "F32";
    case TypeInfo::Pointer: return "Pointer";
    case TypeInfo::Array: return "Array";
    case TypeInfo::Undefined: return "Undefined";
    default: return "Unknown";
    }
}                                     
  
inline bool match_identifier(Token tok, const char* identifier) {
    const u32 len = strlen(identifier);
    return tok.length == len && strncmp(tok.text, identifier, len) == 0;
}

Token consume_until(Lexer* lexer, Token::Type type) {
    while (true) {
        Token tok = lexer->get_token();
        if (tok.type == Token::EndOfSource || tok.type == type) {
            return tok;
        }
    }
}

struct StructInfo {
    const char* name;
    std::vector<StructMemberInfo> members;
};

struct MetaParser {
    Arena str_arena;
    std::vector<StructInfo> struct_info;

    void init();

    void introspect(Lexer* lexer);
    void parse_struct(Lexer* lexer);
    TypeInfo parse_type(Lexer* lexer);
    StructMemberInfo parse_member_declaration(Lexer* lexer);
};

void MetaParser::init() {
    str_arena.init(malloc(megabytes(1)), megabytes(1));
}

TypeInfo MetaParser::parse_type(Lexer* lexer) {
    // optional 'const'
    // typename
    // optional any number of '*'
 
    Token tok;
    tok = lexer->get_token();
    if (tok.type != Token::Identifier) {
        fprintf(out, "Expected type identifier or qualifier\n");
        return { .type = TypeInfo::Undefined, };
    }

    if (match_identifier(tok, "const")) {
        tok = lexer->get_token();
        if (tok.type != Token::Identifier) {
            fprintf(out, "Expected type identifier\n");
            return { .type = TypeInfo::Undefined, };
        }
    }

    if (match_identifier(tok, "struct")) {
        fprintf(out, "Inline struct definitions not currently supported\n");
        parse_struct(lexer);

        return { .type = TypeInfo::Undefined, };
    }

    const Token member_type = tok;

    TypeInfo::Type type;
    if (match_identifier(member_type, "char")) {
        type = TypeInfo::Char;
    } else if (match_identifier(member_type, "u32")) {
        type = TypeInfo::U32;
    } else if (match_identifier(member_type, "f32")) {
        type = TypeInfo::F32;
    }

    u32 pointer_depth = 0;
    tok = lexer->peek_token();
    while (tok.type == Token::Mul) {
        pointer_depth++;
        lexer->get_token();
        tok = lexer->peek_token();
    }

    TypeInfo type_info;
    if (pointer_depth == 0) {
        type_info.type = type;
    } else if (pointer_depth == 1) {
        type_info.type = TypeInfo::Pointer;
        type_info.pointer_type = type;
    } else {
        type_info.type = TypeInfo::Undefined;
    }

    return type_info;
}

StructMemberInfo MetaParser::parse_member_declaration(Lexer* lexer) {
    // type
    //
    // optional fieldname
    // or
    // fieldname, fieldname, ...
    //
    // optional array specification [ const_number ]
    // ;
    
    TypeInfo type_info = parse_type(lexer);

    Token tok;
    tok = lexer->get_token();
    if (tok.type != Token::Identifier) {
        fprintf(out, "Expected identifier name\n");
        return { .name = NULL, };
    }

    const char* member_name = str_arena.push_string("%.*s", tok.length, tok.text);

    i64 array_size = 0;
    tok = lexer->get_token();
    if (tok.type == Token::BracketOpen) {
        tok = lexer->get_token();

        switch (tok.type) {
        case Token::HexValue: 
        case Token::OctValue: 
        case Token::Integer: { 
            array_size = str_to_i64(tok.text, tok.length); 
        } break;
        default: { fprintf(out, "Expected numeric constant for array size\n"); } break;
        }

        consume_until(lexer, Token::BracketClose);
        tok = lexer->get_token();
    }

    if (array_size != 0) {
        type_info.array_size = array_size;
        type_info.array_type = type_info.type;
        type_info.type = TypeInfo::Array;
    }

    if (tok.type != Token::Semicolon) {
        fprintf(out, "Expected semicolon\n");
        return { .name = NULL, };
    }

    return { .name = member_name, .type_info = type_info, .offset = 0 };
}

void MetaParser::parse_struct(Lexer* lexer) {
    // optional name
    // {
    // members list
    // }

    const char* struct_name;

    Token tok;
    tok = lexer->peek_token();
    if (tok.type == Token::Identifier) {
        struct_name = str_arena.push_string("%.*s", tok.length, tok.text);
    } else {
        struct_name = str_arena.push_string("%s", "<anonymous-struct>");
    }

    tok = consume_until(lexer, Token::BraceOpen);
    if (tok.type == Token::EndOfSource) {
        return;
    }

    StructInfo info = { .name = struct_name, };

    while (lexer->peek_token().type != Token::BraceClose) {
        StructMemberInfo member = parse_member_declaration(lexer);
        info.members.push_back(member);
    }

    struct_info.push_back(info);

    lexer->get_token();
}

void MetaParser::introspect(Lexer* lexer) {
    Token tok;
    tok = lexer->get_token();
    if (tok.type == Token::Identifier && (strncmp("struct", tok.text, 6) == 0)) {
        parse_struct(lexer);
    }
}

void parse_file(const char* filename) {
    u64 file_size;
    const char* f = cast(const char*, read_file(filename, &file_size));

    Lexer lexer(f, file_size);

    MetaParser parser;
    parser.init();

    Token tok;
    while (true) {
        tok = lexer.get_token();
        if (tok.type == Token::EndOfSource) {
            break;
        }

        if (match_identifier(tok, "dab_reflect")) {
            parser.introspect(&lexer);
        }
    }

    for (u32 i = 0; i < parser.struct_info.size(); i++) {
        const StructInfo& info = parser.struct_info[i];
        fprintf(out, "constexpr u32 dab_reflect__member_count_%s = %llu;\n", info.name, info.members.size());
    }

    fprintf(out, "\n");

    for (u32 i = 0; i < parser.struct_info.size(); i++) {
        const StructInfo& info = parser.struct_info[i];
        fprintf(out, "constexpr dab::reflect::StructMemberInfo dab_reflect__memberinfo_%s[] = {\n", info.name);
        for (u32 j = 0; j < info.members.size(); j++) {
            const StructMemberInfo& member = info.members[j];
            fprintf(out, "    { .name = \"%s\", ", member.name);
            fprintf(out, ".type_info = { .type = dab::reflect::TypeInfo::%s, ", type_to_string(member.type_info.type));
            if (member.type_info.type == TypeInfo::Pointer) {
                fprintf(out, ".pointer_type = dab::reflect::TypeInfo::%s, ", type_to_string(member.type_info.pointer_type));
            } else if (member.type_info.type == TypeInfo::Array) {
                fprintf(out, ".array_size = %d, .array_type = dab::reflect::TypeInfo::%s, ", 
                        member.type_info.array_size, 
                        type_to_string(member.type_info.array_type));
            }
            fprintf(out, "}, ");
            fprintf(out, ".offset = offsetof(%s, %s) },\n", info.name, member.name);
        }
        fprintf(out, "};\n");
    }
    
    free(cast(void*, f));
}

int main(i32 argc, const char** args) {
    out = fopen("dab_reflect_data.cpp", "wb");

    for (i32 i = 0; i < argc - 1; i++) {
        parse_file(args[1+i]);
    }

    fclose(out);

    return 0;
}
