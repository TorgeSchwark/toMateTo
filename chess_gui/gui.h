// gui.h
#ifndef GUI_H
#define GUI_H

#include <filesystem>
#include <SFML/Graphics.hpp>
#include "toMateTo_engine/move_generation/chess_board.h" // Pfad zu deiner Board-Definition

void init_gui();
bool update_gui(const chess_board& board);

void highlight_moves(const move_stack& moves);

#endif
