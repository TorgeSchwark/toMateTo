#ifndef EVAL
#define EVAL

#include "toMateTo_engine/move_generation/chess_board.h"

int pesto_eval(
    chess_board* chess_board,
    one_side* player,
    one_side* opponent
);

void pesto_init_tables();

#endif