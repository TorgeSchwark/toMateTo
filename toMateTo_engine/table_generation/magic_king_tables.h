#ifndef MAGIC_KING_TABLES
#define MAGIC_KING_TABLES

#include "table_generation/knight_tables.h"
#include "magic_gen.h"

extern Bitboard KING_MOVES_MASK[64];

extern MagicTableEntry PINNED_PIECES_ROOK_MAGIC[64];
extern MagicTableEntry PINNED_PIECES_BISHOP_MAGIC[64];

extern MagicTableEntry ATTACK_PATTERN_ROOK_MAGIC[64];
extern MagicTableEntry ATTACK_PATTERN_BISHOP_MAGIC[64];

extern Bitboard SQUARES_IN_BETWEEN[64][64];
extern Bitboard SQUARES_ON_THE_LINE[64][64];

extern Bitboard DIRECTION_RAYS[64][8];

// Marks squares in between + the to square. No not a line the to field
extern Bitboard CASTLE_FREE[2][2];

extern Bitboard CASTLE_SAVE[2][2];

extern square CASTLE_TO[2][2];

void init_king_mask();

U64 bishop_attacks_patterns_on_the_fly(int sqr, U64 occ);
U64 rook_attacks_patterns_on_the_fly(int sqr, U64 occ);

U64 rook_attacks_on_the_fl_pinned(int sqr, U64 occ);
U64 bishop_attacks_on_the_fly_pinned(int sqr, U64 occ);

void init_pinned_tables_rook_or_bishop(char *piece);
void init_attack_tables_rook_or_bishop(char *piece);

void init_attack_tables_rock_and_bishop();
void init_pinned_tables_rook_and_bishop();

void init_squares_in_between_table();
void init_square_on_the_line_table();

static inline int file_of(int sq) { return sq & 7; }
static inline int rank_of(int sq) { return sq >> 3; }

void init_direction_rays();

#endif


