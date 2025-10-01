#ifndef CHESS_BOARD
#define CHESS_BOARD

#include <cstdint>

#include "move_stack.h"

struct one_side 
{
    int64_t knights;
    int64_t pawns;
    int64_t rooks;
    int64_t bishop;
    int64_t king;
    int64_t queen;
    int64_t side_all;

    void update_side()
    {
        side_all = knights | pawns | rooks | bishop | king | queen;
    }

}

struct chess_board
{
    one_side white;
    one_side black;

    void update_white()
    {
        white.update_side();
    };

    void update_black()
    {
        black.update_black();
    };

    int64_t complete_board;

    void update_board()
    {
        update_black();
        update_white();
        complete_board =  white | black;
    }

    bool whites_turn;

};


#endif 