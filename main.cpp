#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>

#include "chess_gui/gui.h"
#include "toMateTo_engine/move_generation/chess_board.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"

// Here you can add your includes for chess_board, move_stack, your functions, etc.
// #include "chess.h"  (assumed)

int main() {
    chess_board chess_board;
    chess_board.setup_chess_board();

    // initialization
    init_knight_table();

    init_magic_rook_or_bishop("rook");
    init_magic_rook_or_bishop("bishop");
    init_king_mask();
    init_pinned_tables_rook_and_bishop();
    init_attack_tables_rock_and_bishop();

    move_stack move_stack;

    // Example move test
    const int repetitions = 6000000;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        move_stack.clear();
        find_all_moves(&move_stack, &chess_board);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Duration: " << duration.count() << "ms\n";
    move_stack.print_moves();

    // Start GUI
    init_gui();
    highlight_moves(move_stack);

    while (update_gui(chess_board)) {
        // maybe later: input handling, moves, etc.
    }

    return 0;
}
