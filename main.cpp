#include <iostream>
#include <chrono>  // für Zeitmessung
#include "./move_generation/chess_board.h"
#include "./generation/knight_tables.h"
#include "./generation/magic_gen.h"
#include "./generation/king_tables.h"

int main() {
    std::cout << "Willkommen beim Schachspiel!" << std::endl;

    chess_board chess_board;
    chess_board.setup_chess_board();
    init_knight_table();
    chess_board.print_board();
    init_magic_tables();
    init_king_mask();
    init_all_pinned_tables();
    init_all_atack_tables();

    move_stack move_stack;

    // Anzahl der Wiederholungen
    const int repetitions = 66000000; // 800 k moves

    // Zeitmessung starten
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repetitions; ++i) {
        move_stack.clear();  // Annahme: Funktion vorhanden, um vorherige Züge zu löschen
        find_all_moves(&move_stack, &chess_board);
    }

    // Zeitmessung beenden
    auto end_time = std::chrono::high_resolution_clock::now();

    // Dauer berechnen (in Millisekunden)
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Alle Züge wurden " << repetitions << " mal generiert." << std::endl;
    std::cout << "Dauer: " << duration.count() << " Millisekunden." << std::endl;

    // Optional: Einmalige Ausgabe der Züge
    std::cout << "\nBeispielausgabe der letzten Generierung:" << std::endl;
    move_stack.print_moves();

    return 0;
}
