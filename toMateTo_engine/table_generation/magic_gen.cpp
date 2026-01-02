// magic_find.c
// Compile with: gcc -O3 magic_find.c -o magic_find -std=c11
#include "toMateTo_engine/table_generation/magic_gen.h"

#ifndef TABLE_DIR_PATH
#define TABLE_DIR_PATH "./toMateTo_engine/table_generation/tables"
#endif


const char* TABLE_PATH = TABLE_DIR_PATH;


MagicTableEntry BISHOP_MAGIC[64];
MagicTableEntry ROOK_MAGIC[64];


// ------------------ Helpers -------------------
int popcount64(U64 x) {
    return __builtin_popcountll(x);
}

int bit_scan_forward(U64 x) {
    return __builtin_ctzll(x);
}

U64 set_bit(U64 b, int sq) { return b | (1ULL << sq); }
U64 clear_bit(U64 b, int sq) { return b & ~(1ULL << sq); }

// ------------------ Board helpers -------------------
// Files: a=0 ... h=7, ranks 1..8 mapped to 0..7
int sq(int file, int rank) { return rank*8 + file; }

// ------------------ Generate relevant mask for rook (no edges) -------------------
U64 rook_relevant_mask(int s) {
    int rank = s / 8;
    int file = s % 8;
    U64 mask = 0ULL;

    // left
    for (int f = file-1; f > 0; --f) mask |= (1ULL << sq(f, rank));
    // right
    for (int f = file+1; f < 7; ++f) mask |= (1ULL << sq(f, rank));
    // down
    for (int r = rank-1; r > 0; --r) mask |= (1ULL << sq(file, r));
    // up
    for (int r = rank+1; r < 7; ++r) mask |= (1ULL << sq(file, r));

    return mask;
}

// ------------------ Generate relevant mask for bishop (no edges) -------------------
U64 bishop_relevant_mask(int s) {
    int rank = s / 8;
    int file = s % 8;
    U64 mask = 0ULL;

    // NW
    for (int f=file-1, r=rank+1; f>0 && r<7; --f, ++r) mask |= (1ULL << sq(f,r));
    // NE
    for (int f=file+1, r=rank+1; f<7 && r<7; ++f, ++r) mask |= (1ULL << sq(f,r));
    // SW
    for (int f=file-1, r=rank-1; f>0 && r>0; --f, --r) mask |= (1ULL << sq(f,r));
    // SE
    for (int f=file+1, r=rank-1; f<7 && r>0; ++f, --r) mask |= (1ULL << sq(f,r));

    return mask;
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

// ------------------ Compute rook attacks for a square given a full occupancy -------------------
U64 rook_attacks_on_the_fly(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // North (rank increasing)
    for (int rr = r+1; rr <= 7; ++rr) {
        int s = sq(f, rr);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    // South
    for (int rr = r-1; rr >= 0; --rr) {
        int s = sq(f, rr);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    // East
    for (int ff = f+1; ff <= 7; ++ff) {
        int s = sq(ff, r);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    // West
    for (int ff = f-1; ff >= 0; --ff) {
        int s = sq(ff, r);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    return attacks;
}

// ------------------ Compute bishop attacks -------------------
U64 bishop_attacks_on_the_fly(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // NE
    for (int ff = f+1, rr = r+1; ff <= 7 && rr <= 7; ++ff, ++rr) {
        int s = sq(ff, rr);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    // NW
    for (int ff = f-1, rr = r+1; ff >= 0 && rr <= 7; --ff, ++rr) {
        int s = sq(ff, rr);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    // SE
    for (int ff = f+1, rr = r-1; ff <= 7 && rr >= 0; ++ff, --rr) {
        int s = sq(ff, rr);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    // SW
    for (int ff = f-1, rr = r-1; ff >= 0 && rr >= 0; --ff, --rr) {
        int s = sq(ff, rr);
        attacks |= (1ULL<<s);
        if (occ & (1ULL<<s)) break;
    }
    return attacks;
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

bool test_magic_numbers(MagicTableEntry table[64], bool is_rook) {
    for (int square = 0; square < 64; ++square) {
        U64 mask = table[square].mask;
        U64 magic = table[square].magic_number;
        int relevant_bits = table[square].relevant_bits;
        int entries = 1 << relevant_bits;
        U64 *attack_table = table[square].attack_list;

        // Temporäre Arrays
        U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
        if (!occupancies) {
            printf("malloc failed at square %d\n", square);
            return false;
        }

        // Generiere alle Belegungen für dieses Square und Maske
        generate_occupancy_variations(mask, occupancies, entries);

        for (int i = 0; i < entries; ++i) {
            U64 occ = occupancies[i];

            // Berechne Index über Magic
            int index = (int)((occ * magic) >> (64 - relevant_bits));
            U64 expected = attack_table[index];

            // Berechne Attacke "on the fly" (direkt)
            U64 actual = is_rook ? rook_attacks_on_the_fly(square, occ)
                                 : bishop_attacks_on_the_fly(square, occ);

            if (expected != actual) {
                printf("Mismatch at square %d, occupancy %d:\n", square, i);
                printf("Expected: 0x%016lx\n", expected);
                printf("Actual:   0x%016lx\n", actual);
                free(occupancies);
                return false;
            }
        }

        free(occupancies);
    }

    return true;
}

void load_magic_data(const char *filename, MagicTableEntry table[64]) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); printf(filename); return; }

    for (int sq=0; sq < 64; sq++) {
        MagicEntryData entry;
        if (fread(&entry, sizeof(entry), 1, f) != 1) {
            fprintf(stderr, "Error reading MagicEntryData at square %d\n", sq);
            fclose(f);
            exit(1);
        }


        table[sq].magic_number = entry.magic_number;
        table[sq].mask = entry.mask;
        table[sq].relevant_bits = entry.relevant_bits;

        int entries = 1 << entry.relevant_bits;
        table[sq].attack_list = (U64*)malloc(sizeof(U64) * entries);
        if (!table[sq].attack_list) { printf("malloc failed\n"); exit(1); }

        if (fread(table[sq].attack_list, sizeof(U64), entries, f) != (size_t)entries) {
        fprintf(stderr, "Error reading attack_list at square %d\n", sq);
            fclose(f);
            exit(1);
        }

    }

    fclose(f);
}

void save_magic_data(const char *filename, MagicTableEntry table[64]) {
    FILE *f = fopen(filename, "wb");
    if (!f) { perror("save fopen"); return; }

    for (int sq=0; sq < 64; sq++) {
        MagicEntryData entry = {
            .magic_number = table[sq].magic_number,
            .mask = table[sq].mask,
            .relevant_bits = table[sq].relevant_bits,
        };
        fwrite(&entry, sizeof(entry), 1, f);

        // Jetzt attack_list speichern
        int entries = 1 << entry.relevant_bits;
        fwrite(table[sq].attack_list, sizeof(U64), entries, f);
    }

    fclose(f);
    printf("all_saved");
}

void init_all_magics_rooks(char *piece){

    int is_rook = (strcmp(piece, "rook") == 0);
    int magic_done = 0;


    for(int square = 0; square < 64; square++){
        U64 mask = is_rook ? rook_relevant_mask(square) : bishop_relevant_mask(square);
        int relevant_bits = popcount64(mask);
        int entries = 1 << relevant_bits;
        U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
        U64 *attacks = (U64*)malloc(sizeof(U64) * entries);
        U64 *final_attacks = (U64*)malloc(sizeof(U64) * entries);

        if (!occupancies || !attacks) { printf("malloc failed\n"); return; }

        generate_occupancy_variations(mask, occupancies, entries); 

        for (int i=0;i<entries;++i) {
            U64 occ = occupancies[i];
            attacks[i] = is_rook ? rook_attacks_on_the_fly(square, occ) : bishop_attacks_on_the_fly(square, occ);
        }
        U64 magic = 0;
        int ok = find_magic(occupancies, attacks, final_attacks, entries, relevant_bits, &magic, square);
        if (ok) {
            printf("Found magic: 0x%016" PRIx64 " for square %i for rook %i \n", magic, square, is_rook);
            magic_done += 1;

        } else {
            printf("No magic found (try more attempts or different RNG strategy)\n");
        }

        if(is_rook){
            ROOK_MAGIC[square].attack_list = final_attacks;
            ROOK_MAGIC[square].magic_number = magic;
            ROOK_MAGIC[square].mask = mask;
            ROOK_MAGIC[square].relevant_bits = relevant_bits;
        }else{
            BISHOP_MAGIC[square].attack_list = final_attacks;
            BISHOP_MAGIC[square].magic_number = magic;
            BISHOP_MAGIC[square].mask = mask;
            BISHOP_MAGIC[square].relevant_bits = relevant_bits;
        }
        free(attacks);
        free(occupancies);
    }
    if (magic_done == 64) {
        char path[256];

        if (is_rook) {
            snprintf(path, sizeof(path), "%s/magic_table_rook.bin", TABLE_PATH);
            save_magic_data(path, ROOK_MAGIC);
        } else {
            snprintf(path, sizeof(path), "%s/magic_table_bishop.bin", TABLE_PATH);
            save_magic_data(path, BISHOP_MAGIC);
        }

    }
}


void init_magic_tables(){
    char rook_path[256];
    char bishop_path[256];

    snprintf(rook_path, sizeof(rook_path), "%s/magic_table_rook.bin", TABLE_PATH);
    snprintf(bishop_path, sizeof(bishop_path), "%s/magic_table_bishop.bin", TABLE_PATH);

    load_magic_data(rook_path, ROOK_MAGIC);
    load_magic_data(bishop_path, BISHOP_MAGIC);
    if (!test_magic_numbers(ROOK_MAGIC, true)) {
        printf("Fehler in ROOK magic table!\n");
    }else{
        printf("kein Fehler in ROOK magic table!\n");
    }

    if (!test_magic_numbers(BISHOP_MAGIC, false)) {
        printf("Fehler in BISHOP magic table!\n");
    }else{
        printf("kein Fehler in BISHOP magic table!\n");
    }
}

