#include "magic_helper_functions.h"



void change_order(MagicTableEntry* ALREADY_DONE_MAGIC, U64 *blockers, U64 *occupancies, U64 *final_blockers, int relevant_bits, int square){
    int entries = 1 << relevant_bits;
    for(int i = 0; i < entries; i++){
        int index = (occupancies[i] * ALREADY_DONE_MAGIC[square].magic_number) >> (64 - relevant_bits);
        final_blockers[index] = blockers[i];
    }
}

// ------------------ Generate all subsets of mask (map index -> occupancy) -------------------
// positions[]: list of bit indices where mask has bits
void generate_occupancy_variations(U64 mask, U64 *out, int count) {
    int bits = popcount64(mask);
    int pos[64];
    U64 m = mask;
    for (int i=0;i<bits;++i) {
        pos[i] = bit_scan_forward(m); // eg pos= [1,2,3,4,5,6,7,...]
        m &= m - 1;
    }
    // count should be 1<<bits
    int limit = 1<<bits;
    for (int i=0;i<limit;++i) {
        U64 occ = 0ULL;
        //smart shit: check wich bits of i is 1 put bit pos[b] so one bit in the possible occupancies 1.
        for (int b=0;b<bits;++b) if (i & (1<<b)) occ |= (1ULL << pos[b]); 
        out[i] = occ; // all possible occupancies!
    }
}

// ------------------ Random generator (simple xorshift64*) -------------------
uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

U64 generate_candidate(U64 *rng_state, int square) {
    if (square >= 48 && square <= 55) {
        // Für Reihe 7: nicht zu sparse
        return xorshift64star(rng_state) & xorshift64star(rng_state);
    } else {
        // Standard (sparse)
        return xorshift64star(rng_state) & xorshift64star(rng_state) & xorshift64star(rng_state);
    }
    
}

// ------------------ Try find a magic number -------------------
int find_magic(U64 *occupancies, U64 *attacks, U64 *final_atacks, int entries, int relevant_bits, U64 *out_magic, int square) {

    // table size
    int table_size = 1 << relevant_bits;
    U64 *used = (U64*)malloc(sizeof(U64) * table_size);
    if (!used) return 0;

    uint64_t rng_state = (uint64_t)time(NULL) ^ 0x9e3779b97f4a7c15ULL;

    // try many candidates
    for (int attempt = 0; attempt < 1000000000; ++attempt) {
        // create candidate: random 64-bit with many zeros (typical approach: random & random & random)
        U64 cand = generate_candidate(&rng_state, square);

        if (attempt % 100 == 0) {
            cand = xorshift64star(&rng_state); // komplett zufällig
        }

        // some implementations require low-popcount of (cand * mask) or cand to be odd; heuristics:
        if (popcount64((cand * 0x9ddfea08eb382d69ULL) & 0xFF) < 0) continue;

        // clear table sentinel
        for (int i=0;i<table_size;++i) used[i] = 0xFFFFFFFFFFFFFFFFULL;

        int fail = 0;
        for (int i=0;i<entries && !fail;++i) {
            U64 occ = occupancies[i];
            // index = (occ * cand) >> (64 - relevant_bits)
            int index = (int)((occ * cand) >> (64 - relevant_bits));
            if (used[index] == 0xFFFFFFFFFFFFFFFFULL) {
                used[index] = attacks[i];
            } else if (used[index] != attacks[i]) {
                fail = 1;
            }
        }
        if (!fail) {
            for (int i=0;i<entries && !fail;++i) {
                U64 occ = occupancies[i];
                int index = (int)((occ * cand) >> (64 - relevant_bits));
                final_atacks[index] = attacks[i];
            }

            *out_magic = cand;
            free(used);
            return 1;
        }
    }
    free(used);
    return 0;
}

void build_magic(
    int square,
    mask_fn relevant_mask,
    attack_fn attack_on_the_fly,
    MagicTableEntry *dest_table,
    int *magic_done
){
    U64 mask = relevant_mask(square);
    int relevant_bits = popcount64(mask);
    int entries = 1 << relevant_bits;

    U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
    U64 *attacks     = (U64*)malloc(sizeof(U64) * entries);
    U64 *final       = (U64*)malloc(sizeof(U64) * entries);

    if (!occupancies || !attacks || !final) {
        printf("malloc failed\n");
        return;
    }

    generate_occupancy_variations(mask, occupancies, entries);

    for (int i = 0; i < entries; ++i)
        attacks[i] = attack_on_the_fly(square, occupancies[i]);

    U64 magic = 0;
    int ok = find_magic(
        occupancies, attacks, final,
        entries, relevant_bits,
        &magic, square
    );

    if (ok) {
        printf("Found magic: 0x%016" PRIx64 " for square %i\n", magic, square);
        (*magic_done)++;
    } else {
        printf("No magic found (try more attempts or different RNG strategy)\n");
    }

    dest_table[square].attack_list   = final;
    dest_table[square].magic_number  = magic;
    dest_table[square].mask          = mask;
    dest_table[square].relevant_bits = relevant_bits;

    free(attacks);
    free(occupancies);
}

void build_reordered_table(
    int square,
    mask_fn relevant_mask,
    MagicTableEntry *source_magic,
    MagicTableEntry *dest_magic,
    attack_like_fn generator
){
    U64 mask = relevant_mask(square);
    int relevant_bits = popcount64(mask);
    int entries = 1 << relevant_bits;

    U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
    U64 *temp        = (U64*)malloc(sizeof(U64) * entries);
    U64 *final       = (U64*)malloc(sizeof(U64) * entries);

    generate_occupancy_variations(mask, occupancies, entries);

    for (int i=0;i<entries;++i)
        temp[i] = generator(square, occupancies[i]);

    change_order(source_magic, temp, occupancies, final, relevant_bits, square);

    dest_magic[square].attack_list   = final;
    dest_magic[square].magic_number  = source_magic[square].magic_number;
    dest_magic[square].mask          = mask;
    dest_magic[square].relevant_bits = relevant_bits;

    free(temp);
    free(occupancies);
}
