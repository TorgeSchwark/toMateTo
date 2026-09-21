#pragma once
// HalfKP feature indexing, exactly as specified:
//
//   p_idx      = piece_type * 2 + piece_color
//   halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
//
// piece_type in [0,4] (pawn, knight, bishop, rook, queen - kings are not
// features, they select *which* weight table half is used), piece_color in
// {0,1}, king_square/piece_square in [0,63]. That gives
//   64 (piece_square) * 10 (p_idx) * 64 (king_square) = 40960
// which is NUM_FEATURES below.
//
// NNUE is evaluated twice per position, once "from White's point of view"
// and once "from Black's", each with its own accumulator (see
// feature_transformer.hpp). orient()/relative_color() turn an absolute
// (square, color) pair into the pair a given perspective would see: its own
// king is always king_square as-is from its own side, its own pieces are
// always color 0, and the board is rank-mirrored for Black so both
// perspectives share one weight table.

#include <cstdint>
#include <cstddef>

namespace nnue {

enum Color : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4 };

constexpr int NUM_SQUARES = 64;
constexpr int NUM_PIECE_TYPES = 5;   // no king: a king can't be "captured" as a feature
constexpr int NUM_COLORS = 2;
constexpr int NUM_PIECE_INDEX = NUM_PIECE_TYPES * NUM_COLORS; // 10
constexpr std::size_t NUM_FEATURES =
    static_cast<std::size_t>(NUM_SQUARES) * NUM_PIECE_INDEX * NUM_SQUARES; // 40960

// Rank-flips a square for Black's perspective so "my side" always reads the
// board from rank 1 upward, the way White natively does.
constexpr int orient(int square, Color perspective) {
    return perspective == WHITE ? square : (square ^ 56);
}

// "My" pieces are always relative-color 0, "their" pieces are always 1.
constexpr Color relative_color(Color piece_color, Color perspective) {
    return static_cast<Color>(static_cast<int>(piece_color) ^ static_cast<int>(perspective));
}

// The exact index formula the caller specified. All three square/king/color
// arguments must already be expressed relative to one perspective.
constexpr int halfkp_index(int piece_square, int piece_type, int piece_color, int king_square) {
    const int p_idx = piece_type * 2 + piece_color;
    return piece_square + (p_idx + king_square * 10) * 64;
}

// Convenience wrapper taking absolute board coordinates and orienting them
// for `perspective` internally.
inline int feature_index(Color perspective, int piece_square, PieceType piece_type,
                          Color piece_color, int king_square) {
    const int sq  = orient(piece_square, perspective);
    const int ksq = orient(king_square, perspective);
    const Color col = relative_color(piece_color, perspective);
    return halfkp_index(sq, static_cast<int>(piece_type), static_cast<int>(col), ksq);
}

} // namespace nnue
