#ifndef TYPES
#define TYPES

#include <cstdint>

using Bitboard = uint64_t;

enum PieceType: std::int8_t{
    NO_PIECE_TYPE, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
};

#endif 