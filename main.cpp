#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>
#include <map>
#include <cstdint>

#include "chess_gui/gui.h"
#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/testing/debugging.h"
#include "testing/stockfish_perft.h"

// pos 4: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
// pos 5: rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  
// pos 6: r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10
// pos 7: 


int main() {
    chess_board gui_board;
    gui_board.setup_chess_board();

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
    init_rows();
    
    // current degub fen: r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2
    setup_fen_position(gui_board, "1r6/2p5/3p4/1P6/RK3p1k/8/4P1P1/8 w - - 4 3");

    // // perft_debug_depth2_undo("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2");
    // // g2g1b fehlt stockfish hat 45 Nodes ToMateTo Missing
    
   
    // Move moves[256];
    // Move* end = find_all_moves(moves, &gui_board);

    // int num_moves = end - moves;

    // std::cout << "Generated moves: " << num_moves << "\n";

    // for (Move* m = moves; m != end; ++m) {
    //     std::cout << "  "
    //             << m->move_to_string(gui_board.whites_turn)
    //             << "\n";
    // }


    //test_position();
    
 
   find_perft_error("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
    
    
    // chess_board board;

    // board.setup_chess_board();

    // setup_fen_position(
    //     board,
    //     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2"
    // );

    // std::map<std::string, uint64_t> result =
    //     try_all_moves(&board, 2);

    // uint64_t total = 0;

    // std::cout << "\n";
    // std::cout << "========================================\n";
    // std::cout << "DEPTH 2\n";
    // std::cout << "========================================\n";

    // for (const auto& [move, count] : result) {

    //     std::cout
    //         << move
    //         << ": "
    //         << count
    //         << "\n";

    //     total += count;
    // }

    // std::cout << "========================================\n";
    // std::cout << "TOTAL: " << total << "\n";
    // std::cout << "========================================\n";
    // Start GUI
    init_gui();

    while (update_gui(gui_board)) {
        // maybe later: input handling, moves, etc.
    }

    return 0;
}
