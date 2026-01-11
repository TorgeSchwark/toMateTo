#ifndef TYPES
#define TYPES

#include <cstdint>

using Bitboard = uint64_t;

enum { BLACK = 0, WHITE = 1 };
enum { KING_SIDE_INDEX = 0, QUEEN_SIDE_INDEX = 1 };

enum squares : int8_t {
    A1 = 0,  B1,  C1,  D1,  E1,  F1,  G1,  H1,
    A2,      B2,  C2,  D2,  E2,  F2,  G2,  H2,
    A3,      B3,  C3,  D3,  E3,  F3,  G3,  H3,
    A4,      B4,  C4,  D4,  E4,  F4,  G4,  H4,
    A5,      B5,  C5,  D5,  E5,  F5,  G5,  H5,
    A6,      B6,  C6,  D6,  E6,  F6,  G6,  H6,
    A7,      B7,  C7,  D7,  E7,  F7,  G7,  H7,
    A8,      B8,  C8,  D8,  E8,  F8,  G8,  H8,
    SQ_NONE = -1
};


typedef uint64_t U64;

typedef U64 (*attack_like_fn)(int square, U64 occ);
typedef U64 (*attack_fn)(int square, U64 occ);
typedef U64 (*mask_fn)(int square);

enum PieceType: std::int8_t{
    NO_PIECE_TYPE, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
};

constexpr int8_t FORWARD        = 8;
constexpr int8_t FORWARD_LEFT   = 7;
constexpr int8_t FORWARD_RIGHT  = 9;
constexpr int8_t DOUBLE_FORWARD = 16;

enum MoveType {
    NORMAL,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

enum PawnMoveType {
    PUSH1,
    PUSH2,
    CAPL,
    CAPR,
    PROMO_PUSH,
    PROMO_CAPL,
    PROMO_CAPR,
    EPL,
    EPR
};


enum {
    FILE_A, FILE_B, FILE_C, FILE_D,
    FILE_E, FILE_F, FILE_G, FILE_H
};

enum {
    RANK_1, RANK_2, RANK_3, RANK_4,
    RANK_5, RANK_6, RANK_7, RANK_8
};

typedef int8_t square;


#ifndef TABLE_DIR_PATH
#define TABLE_DIR_PATH "./toMateTo_engine/table_generation/tables"
#endif

enum CastlingRights : int8_t {
    NO_CASTLING,
    WHITE_KING_SIDE,
    WHITE_QUEEN_SIDE = WHITE_KING_SIDE << 1,
    BLACK_KING_SIDE  = WHITE_KING_SIDE << 2,
    BLACK_QUEEN_SIDE = WHITE_KING_SIDE << 3,

    KING_SIDE      = WHITE_KING_SIDE | BLACK_KING_SIDE,
    QUEEN_SIDE     = WHITE_QUEEN_SIDE | BLACK_QUEEN_SIDE,
    WHITE_CASTLING = WHITE_KING_SIDE | WHITE_QUEEN_SIDE,
    BLACK_CASTLING = BLACK_KING_SIDE | BLACK_QUEEN_SIDE,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,

    CASTLING_RIGHT_NB = 16
};

struct StateInfo {
    PieceType captured;
    square epSquare;
    CastlingRights castling;
    int rule50;
    // optional: hash, material, etc.
};

typedef uint64_t U64;

typedef struct {
    U64 mask;
    U64 *attack_list;      // array of length num_entries
    U64 magic_number;
    int relevant_bits;
} MagicTableEntry;

// Struct zum Zwischenspeichern ohne Pointer
typedef struct {
    U64 magic_number;
    U64 mask;
    int relevant_bits;
    // attack_list wird separat gespeichert, weil variabel lang
} MagicEntryData;



// Castling Writes like in Stockfish!

inline CastlingRights operator|(CastlingRights a, CastlingRights b) {
    return CastlingRights(int(a) | int(b));
}

inline CastlingRights operator&(CastlingRights a, CastlingRights b) {
    return CastlingRights(int(a) & int(b));
}

inline CastlingRights operator^(CastlingRights a, CastlingRights b) {
    return CastlingRights(int(a) ^ int(b));
}

inline CastlingRights operator~(CastlingRights a) {
    return CastlingRights(~int(a));
}

inline CastlingRights& operator|=(CastlingRights& a, CastlingRights b) {
    return a = a | b;
}

inline CastlingRights& operator&=(CastlingRights& a, CastlingRights b) {
    return a = a & b;
}

#endif 