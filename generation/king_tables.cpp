#include "king_tables.h"

int64_t KING_MOVES_MASK[64];

MagicTableEntry PINNED_PIECES_ROOK_MAGIC[64];
MagicTableEntry PINNED_PIECES_BISHOP_MAGIC[64];

MagicTableEntry ATACK_PATTERN_ROOK_MAGIC[64];
MagicTableEntry ATACK_PATTERN_BISHOP_MAGIC[64];

int directions[8][2] = {{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1}};

void init_king_mask(){
    for(int x = 0; x < 8; x ++){
        for(int y = 0; y < 8; y ++){
            int64_t mask = 0LL;
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

// ------------------ Compute bishop attacks -------------------
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


void change_order(MagicTableEntry* ALREADY_DONE_MAGIC, U64 *blockers, U64 *occupancies, U64 *final_blockers, int relevant_bits, int square){
    int entries = 1 << relevant_bits;
    for(int i = 0; i < entries; i++){
        int index = (occupancies[i] * ALREADY_DONE_MAGIC[square].magic_number) >> (64 - relevant_bits);
        final_blockers[index] = blockers[i];
    }
}

void init_atack_tables(char *piece){
    load_magic_data("./generation/magic_table_rook.bin", ROOOK_MAGIC);
    load_magic_data("./generation/magic_table_bishop.bin", BISHOP_MAGIC);

    int is_rook = (strcmp(piece, "rook") == 0);
    int magic_done = 0;

    for(int square = 0; square < 64; square++){
        U64 mask = is_rook ? rook_relevant_mask(square) : bishop_relevant_mask(square);

        int relevant_bits = popcount64(mask);
        int entries = 1 << relevant_bits;
        U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
        U64 *atack_patterns = (U64*)malloc(sizeof(U64) * entries);
        U64 *final_atack_patterns = (U64*)malloc(sizeof(U64) * entries);

        generate_occupancy_variations(mask, occupancies, entries);

        for (int i=0;i<entries;++i) {
            U64 occ = occupancies[i];
            atack_patterns[i] = is_rook ? rook_attacks_patterns_on_the_fly(square, occ) : bishop_attacks_patterns_on_the_fly(square, occ);
        }   

        is_rook ? change_order(ROOOK_MAGIC, occupancies, atack_patterns, final_atack_patterns, relevant_bits, square) : change_order(BISHOP_MAGIC, occupancies, atack_patterns, final_atack_patterns, relevant_bits, square);

        if(is_rook){
            ATACK_PATTERN_ROOK_MAGIC[square].attack_list = final_atack_patterns;
            ATACK_PATTERN_ROOK_MAGIC[square].magic_number = ROOOK_MAGIC[square].magic_number;
            ATACK_PATTERN_ROOK_MAGIC[square].mask = mask; // meaningless in this case
            ATACK_PATTERN_ROOK_MAGIC[square].relevant_bits = relevant_bits;
        }else{
            ATACK_PATTERN_BISHOP_MAGIC[square].attack_list = final_atack_patterns;
            ATACK_PATTERN_BISHOP_MAGIC[square].magic_number = BISHOP_MAGIC[square].magic_number;
            ATACK_PATTERN_BISHOP_MAGIC[square].mask = mask; // meaningless in this case
            ATACK_PATTERN_BISHOP_MAGIC[square].relevant_bits = relevant_bits;
        }
    }


}

void init_pinned_tables(char *piece){
    load_magic_data("./generation/magic_table_rook.bin", ROOOK_MAGIC);
    load_magic_data("./generation/magic_table_bishop.bin", BISHOP_MAGIC);
    int is_rook = (strcmp(piece, "rook") == 0);
    int magic_done = 0;

    for(int square = 0; square < 64; square++){
        U64 mask = is_rook ? rook_relevant_mask(square) : bishop_relevant_mask(square);

        int relevant_bits = popcount64(mask);
        int entries = 1 << relevant_bits;
        U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
        U64 *blockers = (U64*)malloc(sizeof(U64) * entries);
        U64 *final_blockers = (U64*)malloc(sizeof(U64) * entries);

        generate_occupancy_variations(mask, occupancies, entries);

        for (int i=0;i<entries;++i) {
            U64 occ = occupancies[i];
            blockers[i] = is_rook ? rook_attacks_on_the_fl_pinned(square, occ) : bishop_attacks_on_the_fly_pinned(square, occ);
        }   

        
        is_rook ? change_order(ROOOK_MAGIC, occupancies, blockers, final_blockers, relevant_bits, square) : change_order(BISHOP_MAGIC, occupancies, blockers, final_blockers, relevant_bits, square);
        
        if(is_rook){
            PINNED_PIECES_ROOK_MAGIC[square].attack_list = final_blockers;
            PINNED_PIECES_ROOK_MAGIC[square].magic_number = ROOOK_MAGIC[square].magic_number;
            PINNED_PIECES_ROOK_MAGIC[square].mask = mask; // meaningless in this case
            PINNED_PIECES_ROOK_MAGIC[square].relevant_bits = relevant_bits;
        }else{
            PINNED_PIECES_BISHOP_MAGIC[square].attack_list = final_blockers;
            PINNED_PIECES_BISHOP_MAGIC[square].magic_number = BISHOP_MAGIC[square].magic_number;
            PINNED_PIECES_BISHOP_MAGIC[square].mask = mask; // meaningless in this case
            PINNED_PIECES_BISHOP_MAGIC[square].relevant_bits = relevant_bits;
        }
    }
}

void init_all_atack_tables(){

    init_atack_tables("rook");
    init_atack_tables("not rook");

}

void init_all_pinned_tables(){

    init_pinned_tables("rook");
    init_pinned_tables("not rook");

}






