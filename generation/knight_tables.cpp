#include "knight_tables.h"

int64_t KNIGHT_LOOCKUP_TABLE[64][9];


bool is_on_board(int x, int y, int x_offset, int y_offset){
    return (x+x_offset >= 0 && x+x_offset < 8 && y+y_offset >= 0 && y+y_offset < 8);
}

void generate_knight_tables(int64_t loockup_table[64][9]){
    int board_column_size = 8;
    int directions[8][2] = {{-2,-1},{-1,-2},{-2,1},{1,-2},{2,-1},{-1,2},{1,2},{2,1}};
    for(int x = 0; x < 8; x++){
        for(int y = 0; y < 8; y++){
            int count = 1;
            for(int direction = 0; direction < 8; direction++){
                if (is_on_board(x, y, directions[direction][0], directions[direction][1])){
                    loockup_table[x*board_column_size+y][count] = 1LL << ((x*board_column_size +y) + directions[direction][0]*board_column_size+ directions[direction][1]);
                    count += 1;
                }
            }
            loockup_table[x*board_column_size+y][0] = count;
        }
    }
}

void init_knight_table() {
    generate_knight_tables(KNIGHT_LOOCKUP_TABLE);
}