#ifndef EVAL
#define EVAL

#include "toMateTo_engine/move_generation/chess_board.h"

int pesto_eval(
    chess_board* chess_board,
    one_side* player,
    one_side* opponent
);

void pesto_init_tables();

constexpr int pesto_piece_type(PieceType piece);

int pesto_piece_value(PieceType piece, square sq, bool is_white, int gamePhase,square ep_square);

int pesto_game_phase(chess_board* board);

#endif