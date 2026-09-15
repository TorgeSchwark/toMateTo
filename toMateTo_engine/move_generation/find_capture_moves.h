#ifndef FIND_CAPTURE_MOVES
#define FIND_CAPTURE_MOVES

#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <bit>
#include <map>

#include "toMateTo_engine/move_generation/move_stack.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/move_generation/types.h"
#include "toMateTo_engine/move_generation/chess_board.h"

Move* find_all_capture_moves(Move* moves, chess_board* chess_board);

Move* find_bishop_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* bishop);

Move* find_knight_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy);

Move* find_pawn_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy);

void find_different_pawn_capture_moves(Bitboard pawns, Bitboard empty, one_side* player, one_side* enemy, chess_board* chess_board, Bitboard* results);

Move* find_rook_capture_moves(Move* moves, chess_board* chess_board, one_side* player, one_side* enemy, Bitboard* rook);

#endif