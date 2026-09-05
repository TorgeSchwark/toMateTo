#include "move_gen_speed_testing.h"


const int AMOUNT_TEST_POS = 6;
const std::string FEN_TEST_POSITIONS[AMOUNT_TEST_POS] = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
       "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 
    };

void move_gen_speed_testing(int depth){
    // Returns Moves per seconds for a predefined set of positions FEN 1-6
    std::cout << "\n debug \n" <<  std::flush;

    long long count;
    chess_board chess_board;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(int i = 0; i < AMOUNT_TEST_POS; i++){
        setup_fen_position(chess_board, FEN_TEST_POSITIONS[i]);
        count += try_all_moves_recursive(&chess_board, depth); 
        std::cout << " \n PERFS tested: " << i << std::flush;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    float seconds = std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "\n Moves per second: " << count / seconds << "\n";

}