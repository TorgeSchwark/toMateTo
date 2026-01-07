#include "magic_king_tables.h"

Bitboard KING_MOVES_MASK[64];

MagicTableEntry PINNED_PIECES_ROOK_MAGIC[64];
MagicTableEntry PINNED_PIECES_BISHOP_MAGIC[64];

// normal move table but empty directions are not returnt (no atacker)
MagicTableEntry ATTACK_PATTERN_ROOK_MAGIC[64];
MagicTableEntry ATTACK_PATTERN_BISHOP_MAGIC[64];

Bitboard SQUARES_ON_THE_LINE[64][64];
Bitboard SQUARES_IN_BETWEEN[64][64];

// Marks squares in between + the to square. No not a line the to field
Bitboard CASTLE_FREE[2][2] = {
    { (1ULL<<61) | (1ULL<<62),          (1ULL<<57) | (1ULL<<58) | (1ULL<<59) }, // BLACK
    { (1ULL<<5)  | (1ULL<<6),           (1ULL<<1)  | (1ULL<<2)  | (1ULL<<3)  }  // WHITE
};

Bitboard CASTLE_SAVE[2][2] = {
    { (1ULL<<61) | (1ULL<<62),          (1ULL<<58) | (1ULL<<59) },             // BLACK
    { (1ULL<<5)  | (1ULL<<6),           (1ULL<<2)  | (1ULL<<3)  }              // WHITE
};

square CASTLE_TO[2][2] = {
    { 62,                       58 },                          // BLACK
    { 6,                        2  }                           // WHITE
};

int directions[8][2] = {{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1}};

void init_king_move_mask(){
    for(int x = 0; x < 8; x ++){
        for(int y = 0; y < 8; y ++){
            Bitboard mask = 0LL;
            for(int direction_ind = 0; direction_ind < 8; direction_ind++ ){
                if (is_on_board(x,y,directions[direction_ind][0], directions[direction_ind][1])){
                    mask |= 1LL << ((x+directions[direction_ind][0])*8+(y+directions[direction_ind][1]));
                }
            }
            KING_MOVES_MASK[x*8+y] = mask;
        }
    }
}

U64 rook_attacks_on_the_fl_pinned(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // North
    int count = 0, target = -1;
    for (int rr = r+1; rr <= 7; ++rr) {
        int s = sq(f, rr);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    // South
    count = 0; target = -1;
    for (int rr = r-1; rr >= 0; --rr) {
        int s = sq(f, rr);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    // East
    count = 0; target = -1;
    for (int ff = f+1; ff <= 7; ++ff) {
        int s = sq(ff, r);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    // West
    count = 0; target = -1;
    for (int ff = f-1; ff >= 0; --ff) {
        int s = sq(ff, r);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    return attacks;
}

U64 bishop_attacks_on_the_fly_pinned(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // NE
    int count = 0, target = -1;
    for (int ff = f+1, rr = r+1; ff <= 7 && rr <= 7; ++ff, ++rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    // NW
    count = 0; target = -1;
    for (int ff = f-1, rr = r+1; ff >= 0 && rr <= 7; --ff, ++rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    // SE
    count = 0; target = -1;
    for (int ff = f+1, rr = r-1; ff <= 7 && rr >= 0; ++ff, --rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    // SW
    count = 0; target = -1;
    for (int ff = f-1, rr = r-1; ff >= 0 && rr >= 0; --ff, --rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL << s)) {
            ++count;
            target = s;
            if (count > 1) break;
        }
    }
    if (count == 1) attacks |= (1ULL << target);

    return attacks;
}

U64 rook_attacks_patterns_on_the_fly(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // North (rank increasing)
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int rr = r+1; rr <= 7; ++rr) {
            int s = sq(f, rr);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    // South
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int rr = r-1; rr >= 0; --rr) {
            int s = sq(f, rr);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    // East
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int ff = f+1; ff <= 7; ++ff) {
            int s = sq(ff, r);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    // West
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int ff = f-1; ff >= 0; --ff) {
            int s = sq(ff, r);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    return attacks;
}

U64 bishop_attacks_patterns_on_the_fly(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // NE
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int ff = f+1, rr = r+1; ff <= 7 && rr <= 7; ++ff, ++rr) {
            int s = sq(ff, rr);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    // NW
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int ff = f-1, rr = r+1; ff >= 0 && rr <= 7; --ff, ++rr) {
            int s = sq(ff, rr);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    // SE
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int ff = f+1, rr = r-1; ff <= 7 && rr >= 0; ++ff, --rr) {
            int s = sq(ff, rr);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    // SW
    {
        U64 dir_attacks = 0ULL;
        bool found = false;
        for (int ff = f-1, rr = r-1; ff >= 0 && rr >= 0; --ff, --rr) {
            int s = sq(ff, rr);
            dir_attacks |= (1ULL << s);
            if (occ & (1ULL << s)) {
                found = true;
                break;
            }
        }
        if (found) attacks |= dir_attacks;
    }

    return attacks;
}

void init_king_mask() {
    for (int y = 0; y < 8; y++) {          // rank
        for (int x = 0; x < 8; x++) {      // file
            Bitboard mask = 0ULL;

            for (int d = 0; d < 8; d++) {
                if (is_on_board(x, y, directions[d][0], directions[d][1])) {
                    int nx = x + directions[d][0];
                    int ny = y + directions[d][1];

                    mask |= (1ULL << (ny * 8 + nx));
                }
            }

            KING_MOVES_MASK[y * 8 + x] = mask;
        }
    }
}

void init_attack_tables_rook_or_bishop(char *piece)
{   
    char rook_path[256];
    char bishop_path[256];

    snprintf(rook_path, sizeof(rook_path), "%s/magic_table_rook.bin", TABLE_DIR_PATH);
    snprintf(bishop_path, sizeof(bishop_path), "%s/magic_table_bishop.bin", TABLE_DIR_PATH);

    load_magic_data(rook_path, ROOK_MAGIC);
    load_magic_data(bishop_path, BISHOP_MAGIC);

    int is_rook = (strcmp(piece, "rook") == 0);

    for (int square = 0; square < 64; square++) {
        if (is_rook)
            build_reordered_table(square, rook_relevant_mask,
                                  ROOK_MAGIC, ATTACK_PATTERN_ROOK_MAGIC,
                                  rook_attacks_patterns_on_the_fly);
        else
            build_reordered_table(square, bishop_relevant_mask,
                                  BISHOP_MAGIC, ATTACK_PATTERN_BISHOP_MAGIC,
                                  bishop_attacks_patterns_on_the_fly);
    }
    
}

void init_pinned_tables_rook_or_bishop(char *piece)
{
    char rook_path[256];
    char bishop_path[256];

    snprintf(rook_path, sizeof(rook_path), "%s/magic_table_rook.bin", TABLE_DIR_PATH);
    snprintf(bishop_path, sizeof(bishop_path), "%s/magic_table_bishop.bin", TABLE_DIR_PATH);

    load_magic_data(rook_path, ROOK_MAGIC);
    load_magic_data(bishop_path, BISHOP_MAGIC);

    int is_rook = (strcmp(piece, "rook") == 0);

    for (int square = 0; square < 64; square++) {
        if (is_rook)
            build_reordered_table(square, rook_relevant_mask,
                                  ROOK_MAGIC, PINNED_PIECES_ROOK_MAGIC,
                                  rook_attacks_on_the_fl_pinned);
        else
            build_reordered_table(square, bishop_relevant_mask,
                                  BISHOP_MAGIC, PINNED_PIECES_BISHOP_MAGIC,
                                  bishop_attacks_on_the_fly_pinned);
    }
}

void init_attack_tables_rock_and_bishop(){

    init_attack_tables_rook_or_bishop("rook");
    init_attack_tables_rook_or_bishop("not rook");

}

void init_pinned_tables_rook_and_bishop(){

    init_pinned_tables_rook_or_bishop("rook");
    init_pinned_tables_rook_or_bishop("not rook");

}

void init_squares_in_between_table()
{
    for (int from = 0; from < 64; ++from)
    {
        for (int to = 0; to < 64; ++to)
        {
            if (from == to) {
                SQUARES_IN_BETWEEN[from][to] = 0;
                continue;
            }

            Bitboard bb = 0;

            int fr = rank_of(from);
            int ff = file_of(from);
            int tr = rank_of(to);
            int tf = file_of(to);

            int dr = (tr > fr) - (tr < fr);   // -1, 0, +1
            int df = (tf > ff) - (tf < ff);   // -1, 0, +1

            bool aligned =
                (fr == tr) ||                 
                (ff == tf) ||                    
                (fr - tr == ff - tf) ||          
                (fr - tr == tf - ff);            

            if (!aligned) {
                SQUARES_IN_BETWEEN[from][to] = (1ULL << to);
                continue;
            }

            int sq = from;

            while (true)
            {
                sq += dr * 8 + df;

                bb |= (1ULL << sq);

                if (sq == to)
                    break;
            }

            SQUARES_IN_BETWEEN[from][to] = bb;
        }
    }
}

void init_square_on_the_line_table()
{
    for (int from = 0; from < 64; ++from)
    {
        for (int to = 0; to < 64; ++to)
        {
            Bitboard bb = 0;

            if (from == to) {
                SQUARES_ON_THE_LINE[from][to] = (1ULL << from);
                continue;
            }

            int fr = rank_of(from);
            int ff = file_of(from);
            int tr = rank_of(to);
            int tf = file_of(to);

            // Prüfen, ob sie auf derselben Linie liegen
            bool aligned =
                (fr == tr) ||                    // gleiche Reihe
                (ff == tf) ||                    // gleiche Spalte
                (fr - tr == ff - tf) ||          // Diagonale
                (fr - tr == tf - ff);            // Anti-Diagonale

            if (!aligned) {
                SQUARES_ON_THE_LINE[from][to] = 0;
                continue;
            }

            int dr = (tr > fr) - (tr < fr);   // -1, 0, +1
            int df = (tf > ff) - (tf < ff);   // -1, 0, +1

            // 1️⃣ In Richtung (dr, df) bis zum Rand laufen
            int sq = from;
            while (sq >= 0 && sq < 64)
            {
                bb |= (1ULL << sq);

                int r = rank_of(sq) + dr;
                int f = file_of(sq) + df;
                if (r < 0 || r > 7 || f < 0 || f > 7) break;

                sq = r * 8 + f;
            }

            // 2️⃣ In die Gegenrichtung laufen
            sq = from;
            while (sq >= 0 && sq < 64)
            {
                int r = rank_of(sq) - dr;
                int f = file_of(sq) - df;
                if (r < 0 || r > 7 || f < 0 || f > 7) break;

                sq = r * 8 + f;
                bb |= (1ULL << sq);
            }

            SQUARES_ON_THE_LINE[from][to] = bb;
        }
    }
}

