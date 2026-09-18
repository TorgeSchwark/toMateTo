#ifndef TT
#define TT

#include <cstddef>
#include <cstdint>
#include "toMateTo_engine/move_generation/types.h"
#include "toMateTo_engine/move_generation/move_stack.h"
#include "toMateTo_engine/move_generation/chess_board.h"

enum TTFlag
{
    EXACT,
    LOWERBOUND,
    UPPERBOUND
};


struct TTEntry
{
    uint64_t key = 0;
    Move best_move{};
    int score = 0;
    int depth = -1;
    TTFlag flag = EXACT;
    bool quiescence = false;
};



constexpr std::size_t TT_SIZE = 1 << 21;

// ---------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------

extern uint64_t zobrist_piece[2][6][64];
extern uint64_t zobrist_castling[16];
extern uint64_t zobrist_ep[64];
extern uint64_t zobrist_side;

// ---------------------------------------------------------
// Transposition Table
// ---------------------------------------------------------

extern TTEntry transposition_table[TT_SIZE];

// ---------------------------------------------------------
// Functions
// ---------------------------------------------------------

void init_zobrist();

uint64_t calculate_hash(chess_board* board);

#endif