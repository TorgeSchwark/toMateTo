#ifndef MAGIC_GEN
#define MAGIC_GEN

#include <iostream>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

#include "toMateTo_engine/move_generation/types.h"
#include "toMateTo_engine/table_generation/magic_helper_functions.h"
#include "toMateTo_engine/move_generation/chess_board.h"

extern Bitboard PAWN_ATTACK_LOOKUP_TABLE[2][64];

extern MagicTableEntry BISHOP_MAGIC[64];
extern MagicTableEntry ROOK_MAGIC[64];

int sq(int file, int rank);

U64 rook_relevant_mask(int s);

U64 bishop_relevant_mask(int s);

void generate_occupancy_variations(U64 mask, U64 *out, int count);

U64 rook_attacks_on_the_fly(int sqr, U64 occ);

U64 bishop_attacks_on_the_fly(int sqr, U64 occ);



int find_magic(U64 *occupancies, U64 *attacks, U64 *final_atacks, int entries, int relevant_bits, U64 *out_magic, int square);

bool test_magic_numbers(MagicTableEntry table[64], bool is_rook);

void load_magic_data(const char *filename, MagicTableEntry table[64]);

void save_magic_data(const char *filename, MagicTableEntry table[64]);

void init_magic_rook_or_bishop(char *piece);

void load_magic_tables();

#endif