#include <iostream>
#include <chrono>

#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/table_generation/TT.h"
#include "toMateTo_engine/toMateTo/toMateTo.h"
#include "toMateTo_engine/toMateTo/eval.h"
#include "toMateTo_engine/toMateTo/profiler.h"
#include "testing/engine_match.h"


const int AMOUNT_TEST_POS_MAIN = 6;
const std::string FEN_TEST_POSITIONS_MAIN[AMOUNT_TEST_POS_MAIN] = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
       "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 
    };

int main()
{
    // =====================================================
    // INITIALIZATION
    // =====================================================

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
    init_zobrist();

    

    run_engine_match(2.0, 1500);

    // =====================================================
    // TEST POSITION 2
    // =====================================================

    const std::string fen =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";

    constexpr int depth = 10;

    Profiler::calibrate_cpu_frequency();
    Profiler::reset();

    std::string result =
        alpha_beta_tt_toMateTo(FEN_TEST_POSITIONS_MAIN[0], depth);

    std::cout << "\nBest move: "
              << result
              << "\n";

    Profiler::print();


    return 0;
}

// 1 600 710
// 2 263 926