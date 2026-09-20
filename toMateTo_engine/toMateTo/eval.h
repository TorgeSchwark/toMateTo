#ifndef EVAL
#define EVAL

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/toMateTo/profiler.h"

struct PestoEvalResult
{
    int score;
    int game_phase;
};

PestoEvalResult pesto_eval(
    chess_board* chess_board,
    one_side* white,
    one_side* black
);

void pesto_init_tables();

constexpr int pesto_piece_type(PieceType piece);

int pesto_piece_value(PieceType piece, square sq, bool is_white, int gamePhase,square ep_square);

int pesto_game_phase(chess_board* board);

#endif