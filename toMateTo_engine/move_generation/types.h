#ifndef TYPES
#define TYPES

#include <cstdint>

using Bitboard = uint64_t;
typedef uint64_t U64;

typedef U64 (*attack_like_fn)(int square, U64 occ);
typedef U64 (*attack_fn)(int square, U64 occ);
typedef U64 (*mask_fn)(int square);

enum PieceType: std::int8_t{
    NO_PIECE_TYPE, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
};

#ifndef TABLE_DIR_PATH
#define TABLE_DIR_PATH "./toMateTo_engine/table_generation/tables"
#endif


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

enum MoveType {
    NORMAL,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

#endif 