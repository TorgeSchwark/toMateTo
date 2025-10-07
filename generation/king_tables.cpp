#include "king_tables.h"

int64_t KING_MOVES_MASK[64];



int directions[8][2] = {{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1}};

void init_king_tables(){
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

    // North (rank increasing)
    for (int rr = r+1; rr <= 7; ++rr) {
        int s = sq(f, rr);
        
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    // South
    for (int rr = r-1; rr >= 0; --rr) {
        int s = sq(f, rr);
        
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    // East
    for (int ff = f+1; ff <= 7; ++ff) {
        int s = sq(ff, r);
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    // West
    for (int ff = f-1; ff >= 0; --ff) {
        int s = sq(ff, r);
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    return attacks;
}

// ------------------ Compute bishop attacks -------------------
U64 bishop_attacks_on_the_fly_pinned(int sqr, U64 occ) {
    U64 attacks = 0ULL;
    int r = sqr / 8, f = sqr % 8;

    // NE
    for (int ff = f+1, rr = r+1; ff <= 7 && rr <= 7; ++ff, ++rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    // NW
    for (int ff = f-1, rr = r+1; ff >= 0 && rr <= 7; --ff, ++rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    // SE
    for (int ff = f+1, rr = r-1; ff <= 7 && rr >= 0; ++ff, --rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    // SW
    for (int ff = f-1, rr = r-1; ff >= 0 && rr >= 0; --ff, --rr) {
        int s = sq(ff, rr);
        if (occ & (1ULL<<s)){
            attacks |= (1ULL<<s);
            break;
        } 
    }
    return attacks;
}

void init_pinned_tables(char *piece){
    int is_rook = (strcmp(piece, "rook") == 0);
    int magic_done = 0;

    for(int square = 0; square < 64; square++){
        U64 mask = is_rook ? rook_relevant_mask(square) : bishop_relevant_mask(square);

        int relevant_bits = popcount64(mask);
        int entries = 1 << relevant_bits;
        U64 *occupancies = (U64*)malloc(sizeof(U64) * entries);
        U64 *attacks = (U64*)malloc(sizeof(U64) * entries);
        U64 *final_attacks = (U64*)malloc(sizeof(U64) * entries);

        generate_occupancy_variations(mask, occupancies, entries);

        for (int i=0;i<entries;++i) {
            U64 occ = occupancies[i];
            attacks[i] = is_rook ? rook_attacks_on_the_fl_pinned(square, occ) : bishop_attacks_on_the_fly_pinned(square, occ);
        }



    }


}







