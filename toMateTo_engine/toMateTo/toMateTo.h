#ifndef TOMATETO
#define TOMATETO

#include <string>
#include <chrono>
#include <algorithm>

#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/move_generation/find_capture_moves.h"
#include "toMateTo_engine/toMateTo/eval.h"
#include "toMateTo_engine/move_generation/types.h"
#include "toMateTo_engine/table_generation/TT.h"
#include "profiler.h"

int full_search_eval(chess_board *chess_board, int depth);

std::map<std::string, int> toMateTo(std::string fen_position);

int alpha_beta(chess_board *board, int depth, int alpha, int beta, int root_dist);

std::string alpha_beta_tt_toMateTo(std::string fen_position, double time_limit_seconds);

std::string alpha_beta_toMateTo(std::string fen_position, int depth);

int alpha_beta_old(chess_board* board, int depth, int alpha, int beta);

int mvv_lva_score(chess_board *board, Move m);

int mvv_lva_score_pesto(chess_board* board, Move m, int gamePhase);

void sort_capture_moves(Move* moves, Move* end, chess_board* board, int game_phase);

int quiescence(chess_board *board, int alpha, int beta, int root_dist);


#endif