# toMateTo (to Check Mate Torge)

# Runing the engine
to run the engine Simply run make once. It will take a while for the first time running it, since the magic tables and lookup tables are created. 

# Code documentation
## Move generation 
The Move generation can be found in [move_generation](toMateTo_engine/move_generation/). It is based on a bitboard approach using magic hash tables.

## Magic tables /lookup tables
The move magic tables and lookup tables are generated using the code in [table_generation](toMateTo_engine/table_generation/)

## Chess GUI
the Code for the GUI is provided in [chess_gui](chess_gui/)