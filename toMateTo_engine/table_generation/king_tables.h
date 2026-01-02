#ifndef KING_TABLES
#define KING_TABLES

#include "toMateTo_engine/table_generation/knight_tables.h"
#include "magic_gen.h"

extern Bitboard KING_MOVES_MASK[64];

extern MagicTableEntry PINNED_PIECES_ROOK_MAGIC[64];
extern MagicTableEntry PINNED_PIECES_BISHOP_MAGIC[64];

extern MagicTableEntry ATTACK_PATTERN_ROOK_MAGIC[64];
extern MagicTableEntry ATTACK_PATTERN_BISHOP_MAGIC[64];

void init_king_mask();

U64 bishop_attacks_patterns_on_the_fly(int sqr, U64 occ);
U64 rook_attacks_patterns_on_the_fly(int sqr, U64 occ);

U64 rook_attacks_on_the_fl_pinned(int sqr, U64 occ);
U64 bishop_attacks_on_the_fly_pinned(int sqr, U64 occ);

void change_order(MagicTableEntry* ALREADY_DONE_MAGIC, U64 *blockers, U64 *occupancies, U64 *final_blockers, int relevant_bits, int square);

void init_pinned_tables(char *piece);
void init_attack_tables(char *piece);

void init_all_attack_tables();
void init_all_pinned_tables();

#endif


