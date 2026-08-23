#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>

#include "chess_gui/gui.h"
#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/testing/debugging.h"
#include "testing/stockfish_perft.h"


int main() {
    chess_board chess_board;
    chess_board.setup_chess_board();

    // initialization
    init_knight_table();

    init_magic_rook_or_bishop("rook");
    init_magic_rook_or_bishop("bishop");
    init_squares_in_between_table();
    init_square_on_the_line_table();
    init_king_mask();
    init_pinned_tables_rook_and_bishop();
    init_attack_tables_rock_and_bishop();
    init_pawn_attack_lookup();
    init_direction_rays();
    
    setup_fen_position(chess_board, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2");

    
   
    Move moves[256];
    Move* end = find_all_moves(moves, &chess_board);

    int num_moves = end - moves;

    std::cout << "Generated moves: " << num_moves << "\n";

    for (Move* m = moves; m != end; ++m) {
        std::cout << "  "
                << m->move_to_string(chess_board.whites_turn)
                << "\n";
    }

    
 
    // find_perft_error("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");
    
    

    // Start GUI
    init_gui();

    while (update_gui(chess_board)) {
        // maybe later: input handling, moves, etc.
    }

    return 0;
}
