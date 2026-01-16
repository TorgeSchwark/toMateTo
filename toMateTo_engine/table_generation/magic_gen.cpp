// magic_find.c
// Compile with: gcc -O3 magic_find.c -o magic_find -std=c11
#include "toMateTo_engine/table_generation/magic_gen.h"

Bitboard PAWN_ATTACK_LOOKUP_TABLE[2][64]; 

MagicTableEntry BISHOP_MAGIC[64];
MagicTableEntry ROOK_MAGIC[64];

Bitboard PAWN_ROW[2] = {
    0x00FF000000000000ULL, // Black pawns on rank 7
    0x000000000000FF00ULL  // White pawns on rank 2
};

Bitboard PROMOTION_ROW[2] = {
    0x00000000000000FFULL, // Black promotion on rank 1
    0xFF00000000000000ULL  // White promotion on rank 8
};

Bitboard BOARD_FILE[8] = {
    0x0101010101010101ULL, // FILE_A
    0x0202020202020202ULL, // FILE_B
    0x0404040404040404ULL, // FILE_C
    0x0808080808080808ULL, // FILE_D
    0x1010101010101010ULL, // FILE_E
    0x2020202020202020ULL, // FILE_F
    0x4040404040404040ULL, // FILE_G
    0x8080808080808080ULL  // FILE_H
};

Bitboard BOARD_RANK[8] = {
    0x00000000000000FFULL, // RANK_1
    0x000000000000FF00ULL, // RANK_2
    0x0000000000FF0000ULL, // RANK_3
    0x00000000FF000000ULL, // RANK_4
    0x000000FF00000000ULL, // RANK_5
    0x0000FF0000000000ULL, // RANK_6
    0x00FF000000000000ULL, // RANK_7
    0xFF00000000000000ULL  // RANK_8
};



// ------------------ Helpers -------------------


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

U64 rook_relevant_mask_attack(int s) {
    int rank = s / 8;
    int file = s % 8;
    U64 mask = 0ULL;

    // left
    for (int f = file-1; f >= 0; --f) mask |= (1ULL << sq(f, rank));
    // right
    for (int f = file+1; f <= 7; ++f) mask |= (1ULL << sq(f, rank));
    // down
    for (int r = rank-1; r >= 0; --r) mask |= (1ULL << sq(file, r));
    // up
    for (int r = rank+1; r <= 7; ++r) mask |= (1ULL << sq(file, r));

    return mask;
}


U64 bishop_relevant_mask_attack(int s) {
    int rank = s / 8;
    int file = s % 8;
    U64 mask = 0ULL;

    // NW
    for (int f=file-1, r=rank+1; f>=0 && r<=7; --f, ++r) mask |= (1ULL << sq(f,r));
    // NE
    for (int f=file+1, r=rank+1; f<=7 && r<=7; ++f, ++r) mask |= (1ULL << sq(f,r));
    // SW
    for (int f=file-1, r=rank-1; f>=0 && r>=0; --f, --r) mask |= (1ULL << sq(f,r));
    // SE
    for (int f=file+1, r=rank-1; f<=7 && r>=0; ++f, --r) mask |= (1ULL << sq(f,r));

    return mask;
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

        // Generiere alle Belegungen für dieses Square und Maskes
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
    if (!f) {
        fprintf(stderr, "fopen failed for '%s': ", filename);
        perror("");
        return;
    }

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

void init_magic_rook_or_bishop(const char *piece)
{
    int is_rook = (strcmp(piece, "rook") == 0);
    int magic_done = 0;

    for (int square = 0; square < 64; square++) {
        if (is_rook)
            build_magic(square, rook_relevant_mask, rook_attacks_on_the_fly, ROOK_MAGIC, &magic_done);
        else
            build_magic(square, bishop_relevant_mask, bishop_attacks_on_the_fly, BISHOP_MAGIC, &magic_done);
    }

    if (magic_done == 64) {
        char path[256];

        snprintf(path, sizeof(path),
                 is_rook ? "%s/magic_table_rook.bin"
                         : "%s/magic_table_bishop.bin",
                 TABLE_DIR_PATH);

        save_magic_data(path, is_rook ? ROOK_MAGIC : BISHOP_MAGIC);
    }
}

void load_magic_tables(){
    char rook_path[256];
    char bishop_path[256];

    snprintf(rook_path, sizeof(rook_path), "%s/magic_table_rook.bin", TABLE_DIR_PATH);
    snprintf(bishop_path, sizeof(bishop_path), "%s/magic_table_bishop.bin", TABLE_DIR_PATH);


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

inline int sq_index(int x, int y) {
    return y * 8 + x;
}

void generate_pawn_lookup_table() {

    for (int sq = 0; sq < 64; ++sq) {

        int x = sq % 8;
        int y = sq / 8;

        // Black = 0, White = 1
        PAWN_ATTACK_LOOKUP_TABLE[0][sq] = 0ULL;   // black
        PAWN_ATTACK_LOOKUP_TABLE[1][sq] = 0ULL;   // white

        // ---------- WHITE (moves +1 rank) ----------
        if (is_on_board(x, y, -1, +1)) {
            int t = sq_index(x - 1, y + 1);
            PAWN_ATTACK_LOOKUP_TABLE[1][sq] |= (1ULL << t);
        }
        if (is_on_board(x, y, +1, +1)) {
            int t = sq_index(x + 1, y + 1);
            PAWN_ATTACK_LOOKUP_TABLE[1][sq] |= (1ULL << t);
        }

        // ---------- BLACK (moves -1 rank) ----------
        if (is_on_board(x, y, -1, -1)) {
            int t = sq_index(x - 1, y - 1);
            PAWN_ATTACK_LOOKUP_TABLE[0][sq] |= (1ULL << t);
        }
        if (is_on_board(x, y, +1, -1)) {
            int t = sq_index(x + 1, y - 1);
            PAWN_ATTACK_LOOKUP_TABLE[0][sq] |= (1ULL << t);
        }
    }
}

