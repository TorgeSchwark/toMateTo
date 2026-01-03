#ifndef KNIGHT_TABLES
#define KNIGHT_TABLES

#include <cstdint>

#include "toMateTo_engine/move_generation/types.h"

extern Bitboard KNIGHT_LOOKUP_TABLE[64];

extern Bitboard BISHOP_LOOCKUP_TABLE[64][4][8];


bool is_on_board(int x, int y, int x_offset, int y_offset);

void generate_knight_tables(Bitboard loockup_table[64][9]);

void init_knight_table();

#endif 
