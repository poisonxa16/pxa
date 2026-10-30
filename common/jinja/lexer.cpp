#include "lexer.h"
#include "runtime.h"

#include <cctype>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

// PXA: helpers for the Python string escapes in the lexer (\xHH, \uXXXX, \UXXXXXXXX, octal)
static bool read_hex(const std::string & s, size_t & pos, int n, uint32_t & out) {
    if (pos + (size_t) n > s.size()) {
        return false;
    }
    uint32_t v = 0;
    for (int k = 0; k < n; ++k) {
        const char c = s[pos + k];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return false;
        v = v * 16 + (uint32_t) d;
    }
    pos += n;
    out = v;
    return true;
}

// Code points UTF-8 cannot carry (above U+10FFFF, or a lone UTF-16 surrogate U+D800-U+DFFF from a \u escape
// without its pair) are written as U+FFFD, so the template output always stays valid UTF-8.
static void append_utf8(std::string & out, uint32_t cp) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        cp = 0xFFFD;
    }
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

#define FILENAME "jinja-lexer"

namespace jinja {

static void string_lstrip(std::string & s, const char * chars) {
    size_t start = s.find_first_not_of(chars);
    if (start == std::string::npos) {
        s.clear();
    } else {
        s.erase(0, start);
    }
}

static void string_rstrip(std::string & s, const char * chars) {
    size_t end = s.find_last_not_of(chars);
    if (end == std::string::npos) {
        s.clear();
    } else {
        s.erase(end + 1);
    }
}

lexer_result lexer::tokenize(const std::string & source) {
    std::vector<token> tokens;

    // NOTE: do NOT transform the source string (i.e. preprocessing), as we need to keep
    //       the original character positions for error reporting etc.
    std::string src = source;

    if (source.empty()) {
        return {tokens, src};
    }

    // Normalize \r\n or \r to \n
    for (std::string::size_type pos = 0; (pos = src.find("\r\n", pos)) != std::string::npos; ) {
        src.erase(pos, 1);
        ++pos;
    }
    for (std::string::size_type pos = 0; (pos = src.find("\r", pos)) != std::string::npos; ) {
        src.replace(pos, 1, 1, '\n');
        ++pos;
    }

    // In the default configuration:
    //  - a single trailing newline is stripped if present
    //  - other whitespace (spaces, tabs, newlines etc.) is returned unchanged
    if (source.back() == '\n') {
        src.pop_back();
    }

    size_t pos = 0;
    size_t start_pos = 0;
    size_t curly_bracket_depth = 0;

    using pred = std::function<bool(char)>;
    auto consume_while = [&](const pred & predicate) -> std::string {
        std::string str;
        while (predicate(src[pos])) {
            // check for escape char
            if (src[pos] == '\\') {
                // consume backslash
                ++pos;
                // check for end of input
                if (pos >= src.size()) {
                    throw lexer_exception("unexpected end of input after escape character", source, pos);
                }
                // add escaped char
                char escaped_char = src[pos++];
                auto esc = escape_chars.find(escaped_char);
                if (esc != escape_chars.end()) {
                    str += esc->second;
                    continue;
                }
                // PXA (2026-10-23): the rest of Python's string-literal escapes, which jinja2 templates
                // inherit. Flint's template strips NULs with replace("\x00", ""), and the old lexer threw on \x, so
                // the template failed to load and every chat request aborted the server. A template that parsed
                // before takes none of these branches, so its output is unchanged.
                const int n_hex = escaped_char == 'x' ? 2 : escaped_char == 'u' ? 4 : escaped_char == 'U' ? 8 : 0;
                if (n_hex > 0) {
                    uint32_t cp = 0;
                    if (!read_hex(src, pos, n_hex, cp)) {
                        throw lexer_exception(std::string("truncated \\") + escaped_char + " escape", source, pos);
                    }
                    // a \uD800-\uDBFF high surrogate followed by a \uDC00-\uDFFF low one is one code point
                    if (escaped_char == 'u' && cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= src.size() && src[pos] == '\\' && src[pos + 1] == 'u') {
                        size_t p2 = pos + 2;
                        uint32_t lo = 0;
                        if (read_hex(src, p2, 4, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp  = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            pos = p2;
                        }
                    }
                    append_utf8(str, cp);
                    continue;
                }
                if (escaped_char >= '0' && escaped_char <= '7') {   // octal, up to 3 digits (\0 = NUL)
                    uint32_t cp = (uint32_t) (escaped_char - '0');
                    for (int k = 0; k < 2 && pos < src.size() && src[pos] >= '0' && src[pos] <= '7'; ++k) {
                        cp = cp * 8 + (uint32_t) (src[pos++] - '0');
                    }
                    append_utf8(str, cp);
                    continue;
                }
                if (escaped_char == 'a') {
                    str += '\a';
                    continue;
                }
                // Python keeps an unknown escape as written (backslash + char)
                str += '\\';
                str += escaped_char;
                continue;
            }

            str += src[pos++];
            if (pos > src.size()) {
                throw lexer_exception("unexpected end of input during consume_while", source, pos);
            }
        }
        return str;
    };

    auto consume_numeric = [&]() -> std::string {
        std::string num = consume_while(is_integer);
        if (pos < src.size() && src[pos] == '.' && pos + 1 < src.size() && is_integer(src[pos + 1])) {
            ++pos; // Consume '.'
            std::string frac = consume_while(is_integer);
            num += "." + frac;
        }
        return num;
    };

    auto next_pos_is = [&](std::initializer_list<char> chars, size_t n = 1) -> bool {
        if (pos + n >= src.size()) return false;
        for (char c : chars) {
            if (src[pos + n] == c) return true;
        }
        return false;
    };

    // note: default config for chat template: lstrip_blocks = true, trim_blocks = true

    // text\n[space]{block} --> text\n{block}
    bool opt_lstrip_blocks = true;

    // {block}\n[space]text --> {block}[space]text
    bool opt_trim_blocks = true;

    // options set dynamically based on current/last block
    bool is_lstrip_block = false; // example: {%-
    bool is_rstrip_block = false; // example: -%}

    while (pos < src.size()) {
        start_pos = pos;
        // JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());

        // First, consume all text that is outside of a Jinja statement or expression
        token::type last_token_type = tokens.empty()
                                            ? token::close_statement // initial state
                                            : tokens.back().t;
        if (last_token_type == token::close_statement ||
            last_token_type == token::close_expression ||
            last_token_type == token::comment) {

            bool last_block_can_rm_newline = false;
            is_rstrip_block = false;
            if (pos > 3) {
                char c0 = src[pos - 3];
                char c1 = src[pos - 2];
                char c2 = src[pos - 1];
                // strip if: -[%}#]}text
                is_rstrip_block = c0 == '-'
                                    && (c1 == '%' || c1 == '}' || c1 == '#')
                                    && c2 == '}';
                // match behavior of hf.js: exclude {{ and }} cases, regex: ([#%-]})
                last_block_can_rm_newline = (c1 == '#' || c1 == '%' || c1 == '-') && c2 == '}';
            }

            size_t start = pos;
            size_t end = start;
            while (pos < src.size() &&
                    // Keep going until we hit the next Jinja statement or expression
                    !(
                        src[pos] == '{' &&
                        next_pos_is( {'%', '{', '#'} )
                    )) {
                end = ++pos;
            }

            // equivalent to hf.js code: template.replace(/^[ \t]*({[#%-])/gm, "$1");
            if (opt_lstrip_blocks && src[pos] == '{' && next_pos_is({'%', '#', '-'})) {
                size_t current = end;
                while (current > start) {
                    char c = src[current - 1];
                    if (current == 1) {
                        end = 0; // Trim from the start of the string
                        break;
                    }
                    if (c == '\n') {
                        end = current; // Trim from the start of the line
                        break;
                    }
                    if (!std::isspace(static_cast<unsigned char>(c))) {
                        break; // Found non-whitespace before newline, keep
                    }
                    --current;
                }
            }

            std::string text = src.substr(start, end - start);

            // equivalent to hf.js code: template.replace(/([#%-]})\n/g, "$1");
            if (opt_trim_blocks && last_block_can_rm_newline) {
                if (!text.empty() && text.front() == '\n') {
                    text.erase(text.begin());
                }
            }

            if (is_rstrip_block) {
                // example: {last_block}[space]text
                // doing lstrip on text, effectively rstrip the LAST block
                // JJ_DEBUG("RSTRIP block detected, current text: '%s'", text.c_str());
                string_lstrip(text, " \t\r\n");
            }

            is_lstrip_block = src[pos] == '{' && next_pos_is({'{', '%', '#'}) && next_pos_is({'-'}, 2);
            if (is_lstrip_block) {
                // example: text[space]{current_block}
                // doing rstrip on text, effectively lstrip the CURRENT block
                // JJ_DEBUG("LSTRIP block detected, current text: '%s'", text.c_str());
                string_rstrip(text, " \t\r\n");
            }

            if (!text.empty()) {
                // JJ_DEBUG("consumed text: '%s'", text.c_str());
                tokens.push_back({token::text, text, start_pos});
                continue;
            }
        }

        // Possibly consume a comment
        // TODO: handle lstrip/rstrip for comments? (not important for now)
        if (src[pos] == '{' && next_pos_is( {'#'} )) {
            start_pos = pos;
            pos += 2; // Skip the opening {#
            std::string comment;
            while (!(src[pos] == '#' && next_pos_is( {'}'} ))) {
                if (pos + 2 >= src.size()) {
                    throw lexer_exception("missing end of comment tag", source, pos);
                }
                comment += src[pos++];
            }
            JJ_DEBUG("consumed comment: '%s'", comment.c_str());
            tokens.push_back({token::comment, comment, start_pos});
            pos += 2; // Skip the closing #}
            continue;
        }

        if (src[pos] == '-' && (
                last_token_type == token::open_expression ||
                last_token_type == token::open_statement)
        ) {
            JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());
            pos++; // consume '-' in {%- or {{-
            if (pos >= src.size()) break;
        }

        // Consume (and ignore) all whitespace inside Jinja statements or expressions
        consume_while([](char c) { return std::isspace(static_cast<unsigned char>(c)); });

        if (pos >= src.size()) break;

        char ch = src[pos];

        bool is_closing_block = ch == '-' && next_pos_is( {'%', '}'} );

        // Check for unary operators
        if (!is_closing_block && (ch == '-' || ch == '+')) {
            start_pos = pos;
            token::type last_token_type = tokens.empty() ? token::eof : tokens.back().t;
            if (last_token_type == token::text || last_token_type == token::eof) {
                throw lexer_exception(std::string("unexpected character: ") + ch, source, pos);
            }
            switch (last_token_type) {
                case token::identifier:
                case token::numeric_literal:
                case token::string_literal:
                case token::close_paren:
                case token::close_square_bracket:
                    // Part of a binary operator
                    // a - 1, 1 - 1, true - 1, "apple" - 1, (1) - 1, a[1] - 1
                    // Continue parsing normally
                    break;
                default: {
                    // Is part of a unary operator
                    // (-1), [-1], (1 + -1), not -1, -apple
                    ++pos; // Consume the operator

                    // Check for numbers following the unary operator
                    std::string num = consume_numeric();
                    std::string value = std::string(1, ch) + num;
                    token::type t = num.empty() ? token::unary_operator : token::numeric_literal;
                    // JJ_DEBUG("consumed unary operator or numeric literal: '%s'", value.c_str());
                    tokens.push_back({t, value, start_pos});
                    continue;
                }
            }
        }

        // Try to match one of the tokens in the mapping table
        bool matched = false;
        for (const auto & [seq, typ] : ordered_mapping_table) {
            start_pos = pos;
            // Inside an object literal, don't treat "}}" as expression-end
            if (seq == "}}" && curly_bracket_depth > 0) {
                continue;
            }
            if (pos + seq.size() <= src.size() && src.substr(pos, seq.size()) == seq) {
                tokens.push_back({typ, seq, start_pos});
                if (typ == token::open_expression) {
                    curly_bracket_depth = 0;
                } else if (typ == token::open_curly_bracket) {
                    ++curly_bracket_depth;
                } else if (typ == token::close_curly_bracket) {
                    --curly_bracket_depth;
                }

                pos += seq.size();
                matched = true;
                break; // continue main loop
            }
        }
        if (matched) continue; // continue main loop

        // Strings
        if (ch == '\'' || ch == '"') {
            start_pos = pos;
            ++pos; // Skip opening quote
            std::string str = consume_while([ch](char c) { return c != ch; });
            // JJ_DEBUG("consumed string literal: '%s'", str.c_str());
            tokens.push_back({token::string_literal, str, start_pos});
            ++pos; // Skip closing quote
            continue;
        }

        // Numbers
        if (is_integer(ch)) {
            start_pos = pos;
            std::string num = consume_numeric();
            // JJ_DEBUG("consumed numeric literal: '%s'", num.c_str());
            tokens.push_back({token::numeric_literal, num, start_pos});
            continue;
        }

        // Identifiers
        if (is_word(ch)) {
            start_pos = pos;
            std::string word = consume_while(is_word);
            // JJ_DEBUG("consumed identifier: '%s'", word.c_str());
            tokens.push_back({token::identifier, word, start_pos});
            continue;
        }

        throw lexer_exception(std::string("unexpected character: ") + ch, source, pos);
    }

    return {std::move(tokens), src};
}

} // namespace jinja
