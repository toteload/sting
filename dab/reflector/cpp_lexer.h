#pragma once

struct Token {
    enum Type : i32 {
        Uninitialized = -1,
        Plus,
        Div,
        Minus,
        Mul,
        Modulo,
        Xor,
        Equals,
        LogicalNot,
        GreaterThan,
        BitwiseAnd,
        LesserThan,
        BitwiseOr,
        BitwiseNot,
        Comma,
        Dot,
        Questionmark,
        Colon,
        Semicolon,
        BracketOpen,
        BracketClose,
        BraceOpen,
        BraceClose,
        ParenOpen,
        ParenClose,
        HexValue,
        OctValue,
        HexFloat,
        Integer,
        Float,
        PlusPlus,
        PlusEq,
        DivEq,
        MinusMinus,
        MinusEq,
        MulEq,
        ModuloEq,
        Arrow,
        XorEq,
        LogicalEq,
        NotEq,
        ShiftRightEq,
        ShiftRight,
        GreaterEq,
        ShiftLeftEq,
        ShiftLeft,
        LesserEq,
        BitwiseAndEq,
        LogicalAnd,
        BitwiseOrEq,
        Scope,
        LogicalOr,
        Char,
        String,
        Preprocessor,
        Identifier,
        ParseError,
        Comment,
        EndOfSource
    };

    const char* text;
    u32 length;
    u32 column;
    u32 line;

    Type type;
};  

struct Lexer {
    const char* base;
    const char* end;
    const char* at;

    u32 column;
    u32 line;

    Lexer(const char* source, u32 length);

    Token get_token();
    Token peek_token();

private:
    Token init_token();

    void advance();
    void eat_white_space();

    bool is_start_of_comment();
    bool is_start_of_preprocessor();
    bool is_start_of_identifier();
    bool is_start_of_string();

    Token parse_comment();
    Token parse_preprocessor();
    Token parse_identifier();
    Token parse_string();
    Token parse_float();
    Token parse_integer();
};
