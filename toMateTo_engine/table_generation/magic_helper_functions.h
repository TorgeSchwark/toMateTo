#ifndef MAGIC_HELPER_FUNCTIONS
#define MAGIC_HELPER_FUNCTIONS

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "toMateTo_engine/move_generation/types.h"

inline bool is_on_board(int x, int y, int x_offset, int y_offset) {
    return (x + x_offset >= 0 && x + x_offset < 8 &&
            y + y_offset >= 0 && y + y_offset < 8);
}


void change_order(MagicTableEntry* ALREADY_DONE_MAGIC, U64 *blockers, U64 *occupancies, U64 *final_blockers, int relevant_bits, int square);

uint64_t xorshift64star(uint64_t *state);

void build_magic(
    int square,
    mask_fn relevant_mask,
    attack_fn attack_on_the_fly,
    MagicTableEntry *dest_table,
    int *magic_done
);

void build_reordered_table(
    int square,
    mask_fn relevant_mask,
    MagicTableEntry *source_magic,
    MagicTableEntry *dest_magic,
    attack_like_fn generator
);

void generate_occupancy_variations(U64 mask, U64 *out, int count);

inline int popcount64(U64 x) {
    return __builtin_popcountll(x);
}

inline int bit_scan_forward(U64 x) {
    return __builtin_ctzll(x);
}

U64 generate_candidate(U64 *rng_state, int square);

#endif
