#include <iostream>

// Falls du Klassen oder Funktionen aus chess_board.h oder move_stack.h brauchst:
#include "./move_generation/chess_board.h"
#include "./move_generation/move_stack.h"

int main() {
    std::cout << "Willkommen beim Schachspiel!" << std::endl;

    // Hier könntest du z.B. ein ChessBoard-Objekt erstellen (wenn vorhanden)
    chess_board chess_board;
    chess_board.setup_chess_board();
    chess_board.print_board();
    // Programm-Ende
    return 0;
}
