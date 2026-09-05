#ifndef TOMATETO
#define TOMATETO

#include <string>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/toMateTo/eval.h"

float full_search_eval(chess_board* chess_board, int depth);

std::map<std::string, uint64_t> toMateTo(std::string fen_position);

#endif