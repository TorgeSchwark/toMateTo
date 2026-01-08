#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>

#include "chess_gui/gui.h"
#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"

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

    setup_fen_position(chess_board, "rnbqk1nr/pppppppp/8/1b6/2P5/8/8/5K2 w KQkq - 0 1");


    // Example move test
    const int repetitions =  1; // currently 13000000*32 in that pos per s so 416M/s legal!!
    auto start_time = std::chrono::high_resolution_clock::now();
    Move moves[256];
    Move* result;

    for (int i = 0; i < repetitions; ++i) {
        result = find_all_moves(moves, &chess_board);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // print results
    int count = (result - moves);
    for (int i = 0; i < count; ++i) {
            std::cout << moves[i].move_to_string() << "\n";
        }

    

    std::cout << "Duration: " << duration.count() << "ms\n";
    std::cout << moves[0].move_to_string() << "\n";
    printf("%d moves \n", count);
    

    // Start GUI
    init_gui();

    while (update_gui(chess_board)) {
        // maybe later: input handling, moves, etc.
    }

    return 0;
}
