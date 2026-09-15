#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>
#include <map>
#include <cstdint>

#include "chess_gui/gui.h"
#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/move_generation/find_capture_moves.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/testing/debugging.h"
#include "testing/stockfish_perft.h"
#include "testing/move_gen_speed_testing.h"
#include "toMateTo_engine/toMateTo/toMateTo.h"
#include "toMateTo_engine/toMateTo/eval.h"

// pos 4: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
// pos 5: rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8  
// pos 6: r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10
// pos 7: 

const int AMOUNT_TEST_POS_MAIN = 7;
const std::string FEN_TEST_POSITIONS_MAIN[AMOUNT_TEST_POS_MAIN] = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
       "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", "8/8/1Q3q2/8/8/5k2/2K5/8 b - - 0 10",
    };

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
    init_pawn_attack_lookup();
    init_direction_rays();
    init_rows();
    pesto_init_tables();
        
    // 4510484
    // 780618
    // 826245
    
    int depth = 4;

    auto start = std::chrono::high_resolution_clock::now();

    std::map<std::string, int> result =
        alpha_beta_toMaTo(FEN_TEST_POSITIONS_MAIN[6], depth);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time: " << duration.count() << " us\n";

    std::cout << "Alpha-Beta evaluations:\n";

    for (const auto& [move, eval] : result)
    {
        std::cout << move << " : " << eval << '\n';
    }



    for (int i = 0; i < AMOUNT_TEST_POS_MAIN; ++i)
    {
        std::cout << "\n========== Perft Position " << i + 1 << " ==========\n";
        std::cout << FEN_TEST_POSITIONS_MAIN[i] << "\n\n";

        find_perft_error(FEN_TEST_POSITIONS_MAIN[i]);
    }




    // Start GUI
    init_gui();

    while (update_gui(gui_board)) {
        // maybe later: input handling, moves, etc.
    }

    return 0;
}
