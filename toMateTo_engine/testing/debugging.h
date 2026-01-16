#ifndef DEBUGGING
#define DEBUGGING

#include <string>
#include <sys/types.h> // pid_t
#include "toMateTo_engine/move_generation/chess_board.h"

struct Process {
    int in_fd;   // write to child (stdin)
    int out_fd;  // read from child (stdout)
    pid_t pid;
};


void perft_debugging(const std::string& fen, int depth);

int stockfish_move_count(Process& sf, const std::string& fen);

Process start_stockfish(const char* path);


#endif 