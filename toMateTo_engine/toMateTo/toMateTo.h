#ifndef TOMATETO
#define TOMATETO

#include <string>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/move_generation/find_capture_moves.h"
#include "toMateTo_engine/toMateTo/eval.h"
#include "toMateTo_engine/move_generation/types.h"

int full_search_eval(chess_board *chess_board, int depth);

std::map<std::string, int> toMateTo(std::string fen_position);

std::map<std::string, int> alpha_beta_toMaTo(std::string fen_position, int depth);

int alpha_beta(chess_board *board, int depth, int alpha, int beta);

int mvv_lva_score(chess_board *board, Move m);

int mvv_lva_score_pesto(chess_board* board, Move m, int gamePhase);

void sort_capture_moves(Move *moves, Move *end, chess_board *board);

int quiescence(chess_board *board, int alpha, int beta);

#endif