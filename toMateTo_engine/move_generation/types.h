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

#endif 