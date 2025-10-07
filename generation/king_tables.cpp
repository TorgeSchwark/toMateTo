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


