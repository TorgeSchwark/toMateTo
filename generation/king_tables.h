#ifndef KING_TABLES
#define KING_TABLES

#include "knight_tables.h"
#include "magic_gen.h"

extern int64_t KING_MOVES_MASK[64];
extern MagicTableEntry PINNED_PIECES_ROOK_MAGIC[64];
extern MagicTableEntry PINNED_PIECES_BISHOP_MAGIC[64];


void init_king_mask();
void rook_attacks_on_the_fl_pinned();
U64 bishop_attacks_on_the_fly_pinned(int sqr, U64 occ);
void change_order(MagicTableEntry* ALREADY_DONE_MAGIC, U64 *blockers, U64 *occupancies, U64 *final_blockers, int relevant_bits, int square);
void init_pinned_tables(char *piece);
void init_all_pinned_tables();

#endif


