#include "cpp_lexer.h"

// TODO
// numeric literals with suffixes
// binary literals (with 0b prefix)
 
inline bool Lexer::is_start_of_comment() {
    if ((end - at) >= 2 && (at[0] == '/' && (at[1] == '/' || at[1] == '*'))) {
        return true;
    } else {
        return false;
    }
}
 
inline bool Lexer::is_start_of_preprocessor() {
    return (*at == '#');
}
         
inline bool Lexer::is_start_of_identifier() {
    return (is_alpha(*at) || (*at) == '_');
}
 
inline bool Lexer::is_start_of_string() {
    return (*at == '"' || *at == '\'');
}
 
inline void Lexer::advance() {
    if (*at == '\n') {
        line++;
        column = 1;
    } else {
        column++;
    }
    at++;
}                                                 

void Lexer::eat_white_space() {
    while (at != end && is_whitespace(*at)) {
        advance();
    }
}

Token Lexer::parse_comment() {
    Token token = init_token();
    token.type = Token::Comment;

    if ((end - at) >= 2) {
        if (at[1] == '/') {
            // Single line comment
            while (at != end && *at != '\r' && *at != '\n') {
                advance();
            }
            token.length = (u32)(at - token.text);
        } else {
            // Multi line comment
            while (at != end && (at[-1] != '*' || at[0] != '/')) {
                advance();
            }
            // The source ended, but the comment was not closed
            if (at == end) {
                token.type = Token::ParseError;
                return token;
            }
            advance();
            token.length = (u32)(at - token.text);
        }
    }

    return token;
}

Token Lexer::parse_preprocessor() {
    Token token = init_token();
    token.type = Token::Preprocessor;

    while (at != end && *at != '\r' && *at != '\n') {
        advance();
    }

    token.length = (u32)(at - token.text);
    return token;
}

Token Lexer::parse_identifier() {
    Token token = init_token();
    token.type = Token::Identifier;

    const char* start = at;

    while (at != end &&
           (is_alpha(*at) || is_digit(*at) || *at == '_')) {
        advance();
    }

    token.length = (u32)(at - start);

    return token; 
}

Token Lexer::parse_string() {
    Token token = init_token();

    char delimiter = *at;
    advance();

    if (delimiter == '\'') {
        token.type = Token::Char;
    } else {
        token.type = Token::String;
    }

    while (at != end) {
        if (at[0] == delimiter) { 
            if (at[-1] == '\\') {
                if (at[-2] == '\\') {
                    break;
                }
                advance();
                continue;
            }
            break;
        }
        advance();
    }

    if (at == end) {
        token.type = Token::ParseError;
        return token;
    }

    advance();
    token.length = (u32)(at - token.text);

    return token;
}

Token Lexer::parse_float() {
    Token token = init_token();
    token.type = Token::Float;

    const char* start = at;
    while (at != end && is_digit(*at)) {
        advance();
    }

    if (*at == '.') {
        advance();
        while (at != end && is_digit(*at)) {
            advance();
        }
    }

    if (*at == 'e' || *at == 'E' ||
        *at == 'p' || *at == 'P') {
        advance();
        if (at != end && (*at == '-' || *at == '+')) {
            advance();
        }
        while (at != end && is_digit(*at)) {
            advance();
        }
    }

    if (*at == 'f' || *at == 'F') {
        advance();
    }

    token.length = (u32)(at - start);

    return token;
}

Lexer::Lexer(const char* source, u32 length) :
    base(source),
    end(source + length),
    at(source),
    column(1),
    line(1)
{ }
 
Token Lexer::init_token() {
    return { .text = at, .column = column, .line = line, .type = Token::Uninitialized };
}

Token Lexer::get_token() {
    eat_white_space();

    Token token = init_token();

    if (at == end) {
        token.type = Token::EndOfSource;
        return token;
    }

    // arithmetic operators:
    // + - / * % = ++ --
    // comparison operators:
    // == != > < >= <=
    // logical operators:
    // ! && ||
    // bitwise operators:
    // ~ & | ^ >> <<
    // compound assignment operators:
    // += -= *= /= %= &= |= ^= <<= >>=
    // member and pointer operators:
    // [ ] * & . ->
    // other operators:
    // , ? (ternary cond)
    // brace types
    // [ ] { } ( )

    if (is_start_of_comment()) { return parse_comment(); }
    if (is_start_of_preprocessor()) { return parse_preprocessor(); }

    // TODO: check for keywords in identifier
    if (is_start_of_identifier()) { return parse_identifier(); }
    if (is_start_of_string()) { return parse_string(); }

    switch (*at) {
    case '0': {
        // TODO: finish this; octal, hex float parsing
        const char* start = at;
        // Could be octal constant or hex constant or float constant
        if (end - at >= 2 && (at[1] == 'x' || at[1] == 'X')) {
            // It is a hexadecimal constant
            // Could be integer hexadecimal or float hexadecimal
            at += 2;
            while (at != end && is_hex(*at)) {
                advance();
            }

            if (at == end) {
                token.type = Token::HexValue;
                token.length = (u32)(at - start);
                return token;
            }

            if (*at == '.') {
                // We have a hex float literal
                advance();
                while (at != end && is_hex(*at)) {
                    advance();
                }

                if (at == end) {
                    token.type = Token::HexFloat;
                    token.length = (u32)(at - start);
                    return token;
                }

                // float exponent symbols: e, E, p, P
                if (*at == 'e' || *at == 'E' || *at == 'p' || *at == 'P') {
                    if (end - at >= 1 && (at[1] == '-' || at[1] == '+')) {
                        at += 2;
                    }

                    while (at != end && is_digit(*at)) {
                        advance();
                    }

                    // Float suffixes: f, F, l, L
                    if (*at == 'f' || *at == 'F' || *at == 'l' || *at == 'L') {
                        advance();
                    }

                    token.type = Token::HexFloat;
                    token.length = (u32)(at - start);
                    return token;
                }
            }

            token.type = Token::HexValue;
            token.length = (u32)(at - start);
            return token;
        }

        // Could be octal or float
        const char* lookahead = at;
        while (lookahead != end && is_digit(*lookahead)) {
            lookahead++;
        }

        if (*lookahead == '.' || *lookahead == 'e' || *lookahead == 'E' ||
            *lookahead == 'p' || *lookahead == 'P') {
            // We have a float on our hands
            return parse_float();
        }

        // Octal constant
        advance();
        while (at != end && is_octal(*at)) {
            advance();
        }

        token.type = Token::OctValue;
        token.length = (u32)(at - start);
        return token;
    }
    case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9': {
        // Could be a float or an integer
        const char* lookahead = at + 1;
        while (lookahead != end && is_digit(*lookahead)) {
            lookahead++;
        }

        if (*lookahead == '.' || *lookahead == 'e' || *lookahead == 'E' ||
            *lookahead == 'p' || *lookahead == 'P') {
            // We have a float on our hands
            return parse_float();
        }

        token.type = Token::Integer;
        token.length = (u32)(lookahead - at);
        at = lookahead;
        return token;
    }
    case '.': {
        // Could be operator . or float literal
        if (end - at >= 2) {
            if (is_digit(at[1])) {
                return parse_float();
            }
        }

        token.type = Token::Dot;
        token.length = 1;
        advance();
        return token;
    }
    case '+': {
        // Could be + or ++ or +=
        if (end - at >= 2) {
            if (at[1] == '+') {
                token.type = Token::PlusPlus;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '=') {
                token.type = Token::PlusEq;
                token.length = 2;
                at += 2;
                return token;
            }
        }

        token.type = Token::Plus;
        token.length = 1;
        advance();
        return token;
    }
    case '-': {
        // Could be - or -- or -= or ->
        if (end - at >= 2) {
            if (at[1] == '-') {
                token.type = Token::MinusMinus;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '=') {
                token.type = Token::MinusEq;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '>') {
                token.type = Token::Arrow;
                token.length = 2;
                at += 2;
                return token;
            }
        }

        token.type = Token::Minus;
        token.length = 1;
        advance();
        return token;
    }
    case '/': {
        // Could be / or /=
        if (end - at >= 2 && at[1] == '=') {
            token.type = Token::DivEq;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::Div;
        token.length = 1;
        advance();
        return token;
    }
    case '*': {
        // Could be * or *=
        if (end - at >= 2 && at[1] == '=') {
            token.type = Token::MulEq;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::Mul;
        token.length = 1;
        advance();
        return token;
    }
    case '%': {
        // Could be % or %=
        if (end - at >= 2 && at[1] == '=') {
            token.type = Token::ModuloEq;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::Modulo;
        token.length = 1;
        advance();
        return token;
    }
    case '=': {
        // Could be = or ==
        if (end - at >= 2 && at[1] == '=') {
            token.type = Token::LogicalEq;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::Equals;
        token.length = 1;
        advance();
        return token;
    }
    case '!': {
        // Could be ! or !=
        if (end - at >= 2 && at[1] == '=') {
            token.type = Token::NotEq;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::LogicalNot;
        token.length = 1;
        advance();
        return token;
    }
    case '^': {
        // Could be ^ or ^=
        if (end - at >= 2 && at[1] == '=') {
            token.type = Token::XorEq;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::Xor;
        token.length = 1;
        advance();
        return token;
    }
    case '>': {
        // Could be > or >= or >> or >>=
        if (end - at >= 3 && at[1] == '>' && at[2] == '=') {
            token.type = Token::ShiftRightEq;
            token.length = 3;
            at += 3;
            return token;
        }
        if (end - at >= 2) {
            if (at[1] == '=') {
                token.type = Token::GreaterEq;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '>') {
                token.type = Token::ShiftRight;
                token.length = 2;
                at += 2;
                return token;
            }
        }
        token.type = Token::GreaterThan;
        token.length = 1;
        advance();
        return token;
    }
    case '<': {
        // Could be < or <= or << or >>=
        if (end - at >= 3 && at[1] == '<' && at[2] == '=') {
            token.type = Token::ShiftLeftEq;
            token.length = 3;
            at += 3;
            return token;
        }
        if (end - at >= 2) {
            if (at[1] == '=') {
                token.type = Token::LesserEq;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '>') {
                token.type = Token::ShiftLeft;
                token.length = 2;
                at += 2;
                return token;
            }
        }
        token.type = Token::LesserThan;
        token.length = 1;
        advance();
        return token;
    }
    case '&': {
        // Could be & or &= or &&
        if (end - at >= 2) {
            if (at[1] == '=') {
                token.type = Token::BitwiseAndEq;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '&') {
                token.type = Token::LogicalAnd;
                token.length = 2;
                at += 2;
                return token;
            }
        }
        token.type = Token::BitwiseAnd;
        token.length = 1;
        advance();
        return token;
    }
    case '|': {
        // Could be | or |= or ||
        if (end - at >= 2) {
            if (at[1] == '=') {
                token.type = Token::BitwiseOrEq;
                token.length = 2;
                at += 2;
                return token;
            }
            if (at[1] == '|') {
                token.type = Token::LogicalOr;
                token.length = 2;
                at += 2;
                return token;
            }
        }
        token.type = Token::BitwiseOr;
        token.length = 1;
        advance();
        return token;
    }
    case ':': {
        // Could be : or ::
        if (end - at >= 2 && at[1] == ':') {
            token.type = Token::Scope;
            token.length = 2;
            at += 2;
            return token;
        }

        token.type = Token::Colon;
        token.length = 1;
        advance();
        return token;
    }
    case '~': {
        // bitwise not
        token.type = Token::BitwiseNot;
        token.length = 1;
        advance();
        return token;
    }
    case ',': {
        // comma
        token.type = Token::Comma;
        token.length = 1;
        advance();
        return token;
    }
    case '?': {
        // ? of ternary operator
        token.type = Token::Questionmark;
        token.length = 1;
        advance();
        return token;
    }
    case ';': { 
        token.type = Token::Semicolon;
        token.length = 1;
        advance();
        return token;
    }
    case '[': { 
        token.type = Token::BracketOpen;
        token.length = 1;
        advance();
        return token;
    }
    case ']': { 
        token.type = Token::BracketClose;
        token.length = 1;
        advance();
        return token;
    }
    case '{': { 
        token.type = Token::BraceOpen;
        token.length = 1;
        advance();
        return token;
    }
    case '}': { 
        token.type = Token::BraceClose;
        token.length = 1;
        advance();
        return token;
    }
    case '(': { 
        token.type = Token::ParenOpen;
        token.length = 1;
        advance();
        return token;
    }
    case ')': { 
        token.type = Token::ParenClose;
        token.length = 1;
        advance();
        return token;
    }
    }

    return token;
}

Token Lexer::peek_token() {
    Lexer tmp = *this;
    Token tok = get_token();
    *this = tmp;
    return tok;
}
