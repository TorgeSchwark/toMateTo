#ifndef KNIGHT_TABLES
#define KNIGHT_TABLES

#include <cstdint>

extern int64_t KNIGHT_LOOKUP_TABLE[64];

extern int64_t BISHOP_LOOCKUP_TABLE[64][4][8];


bool is_on_board(int x, int y, int x_offset, int y_offset);

void generate_knight_tables(int64_t loockup_table[64][9]);

void init_knight_table();

#endif 
