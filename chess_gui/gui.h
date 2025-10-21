// gui.h
#ifndef GUI_H
#define GUI_H

#include <SFML/Graphics.hpp>
#include "../move_generation/chess_board.h" // Pfad zu deiner Board-Definition

void init_gui();
bool update_gui(const chess_board& board);

#endif
