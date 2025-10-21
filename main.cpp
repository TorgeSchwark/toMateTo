#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>

#include "./chess_gui/gui.h"
#include "./move_generation/chess_board.h"
#include "./generation/knight_tables.h"
#include "./generation/magic_gen.h"
#include "./generation/king_tables.h"



// Hier deine includes für chess_board, move_stack, deine Funktionen etc.
// #include "chess.h"  (angenommen)


int main() {
    chess_board chess_board;
    chess_board.setup_chess_board();

    // init
    init_knight_table();
    init_magic_tables();
    init_king_mask();
    init_all_pinned_tables();
    init_all_atack_tables();

    move_stack move_stack;

    // Beispielhafter Move-Test
    const int repetitions = 1000000;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        move_stack.clear();
        find_all_moves(&move_stack, &chess_board);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Dauer: " << duration.count() << "ms\n";
    move_stack.print_moves();

    // GUI starten
    init_gui();
    while (update_gui(chess_board)) {
        // ggf. später: input handling, züge etc.
    }


    return 0;
}
