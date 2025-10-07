#ifndef MAGIC_GEN
#define MAGIC_GEN

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

typedef uint64_t U64;


typedef struct {
    U64 mask;
    U64 *attack_list;      // array of length num_entries
    U64 magic_number;
    int relevant_bits;
} MagicTableEntry;

// Struct zum Zwischenspeichern ohne Pointer
typedef struct {
    U64 magic_number;
    U64 mask;
    int relevant_bits;
    // attack_list wird separat gespeichert, weil variabel lang
} MagicEntryData;

extern MagicTableEntry BISHOP_MAGIC[64];
extern MagicTableEntry ROOOK_MAGIC[64];

U64 rook_relevant_mask(int s);

U64 bishop_relevant_mask(int s);

void generate_occupancy_variations(U64 mask, U64 *out, int count);

U64 rook_attacks_on_the_fly(int sqr, U64 occ);

U64 bishop_attacks_on_the_fly(int sqr, U64 occ);

uint64_t xorshift64star(uint64_t *state);

U64 generate_candidate(U64 *rng_state, int square);

int find_magic(U64 *occupancies, U64 *attacks, U64 *final_atacks, int entries, int relevant_bits, U64 *out_magic, int square);

bool test_magic_numbers(MagicTableEntry table[64], bool is_rook);

void load_magic_data(const char *filename, MagicTableEntry table[64]);

void save_magic_data(const char *filename, MagicTableEntry table[64]);

void init_all_magics_rooks(char *piece);

void init_magic_tables();

#endif