#ifndef MAGIC_KING_TABLES
#define MAGIC_KING_TABLES

#include "table_generation/knight_tables.h"
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

void init_pinned_tables_rook_or_bishop(char *piece);
void init_attack_tables_rook_or_bishop(char *piece);

void init_attack_tables_rock_and_bishop();
void init_pinned_tables_rook_and_bishop();

#endif


