#ifndef KNIGHT_TABLES
#define KNIGHT_TABLES

#include <cstdint>

#include "toMateTo_engine/move_generation/types.h"
#include "toMateTo_engine/move_generation/chess_board.h"


extern Bitboard KNIGHT_LOOKUP_TABLE[64];

extern Bitboard BISHOP_LOOCKUP_TABLE[64][4][8];

void generate_knight_tables(Bitboard loockup_table[64][9]);

void init_knight_table();

#endif 
