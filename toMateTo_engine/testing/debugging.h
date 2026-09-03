#ifndef DEBUGGING
#define DEBUGGING

#include <unistd.h>     // fork, pipe, read, write, dup2, close
#include <sys/types.h>  // pid_t
#include <sys/wait.h>   // waitpid (optional, aber empfohlen)
#include <cstdio>      // printf, sscanf
#include <cstring>     // strstr

#include "toMateTo_engine/move_generation/chess_board.h"

// -------------------------------
// Stockfish process wrapper
// -------------------------------
struct Process {
    int in_fd;     // write to Stockfish stdin
    int out_fd;    // read from Stockfish stdout
    pid_t pid;     // Stockfish process id
};

Process start_stockfish(const char* path);

// -------------------------------
// Stockfish control
// -------------------------------

// Forks and execs Stockfish, performs UCI handshake (uci / uciok)
Process start_and_init_stockfish(const char* path);

// Returns number of legal moves in given position (perft 1)
int stockfish_move_count(Process& sf, const std::string& fen);

// -------------------------------
// Perft debugging
// -------------------------------

// Entry point: compares engine vs Stockfish recursively up to depth
void perft_debugging(const std::string& fen, int depth);

// Internal recursive perft debugger
void perft_debug_recursive(
    class chess_board& board,
    int depth,
    Process& sf
);

void perft_debug_depth2_undo(const std::string& fen);

#endif 