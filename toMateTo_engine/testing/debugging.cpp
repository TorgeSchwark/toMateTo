#include "debugging.h"


Process start_stockfish(const char* path) {
    int in_pipe[2];
    int out_pipe[2];

    if (pipe(in_pipe) == -1) {
        perror("pipe in");
        exit(1);
    }

    if (pipe(out_pipe) == -1) {
        perror("pipe out");
        exit(1);
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(1);
    }

    if (pid == 0) {
        // Child: Stockfish
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);

        close(in_pipe[1]);
        close(out_pipe[0]);

        execl(path, path, nullptr);
        _exit(1); // exec failed
    }

    // Parent
    close(in_pipe[0]);
    close(out_pipe[1]);

    return Process{
        .in_fd  = in_pipe[1],
        .out_fd = out_pipe[0],
        .pid    = pid
    };
}


int stockfish_move_count(Process& sf, const std::string& fen) {
    char buf[256];
    std::string acc;

    dprintf(sf.in_fd, "position fen %s\n", fen.c_str());
    dprintf(sf.in_fd, "go perft 1\n");

    while (true) {
        ssize_t n = read(sf.out_fd, buf, sizeof(buf));
        if (n <= 0)
            return -1;

        acc.append(buf, n);

        size_t pos;
        while ((pos = acc.find('\n')) != std::string::npos) {
            std::string line = acc.substr(0, pos);
            acc.erase(0, pos + 1);

            if (line.rfind("Nodes searched:", 0) == 0) {
                int nodes;
                sscanf(line.c_str(), "Nodes searched: %d", &nodes);
                return nodes;
            }
        }
    }
}



Process start_and_init_stockfish() {
    auto sf = start_stockfish("stockfish");

    dprintf(sf.in_fd, "uci\n");

    char buf[256];
    std::string acc;
    while (true) {
        ssize_t n = read(sf.out_fd, buf, sizeof(buf));
        acc.append(buf, n);
        if (acc.find("uciok") != std::string::npos)
            break;
    }

    return sf;
}


void perft_debugging(const std::string& fen, int depth) {
    Process sf = start_and_init_stockfish();

    chess_board board;
    board.setup_chess_board();
    setup_fen_position(board, fen);

    perft_debug_recursive(board, depth, sf);
}



void perft_debug_recursive(
    chess_board& board,
    int depth,
    Process& sf
) {
    Move moves[256];
    Move* end = find_all_moves(moves, &board);
    int engine_moves = end - moves;

    std::string fen = board_to_fen(board);
    int sf_moves = stockfish_move_count(sf, fen);

    if (engine_moves != sf_moves) {
        fprintf(stderr, "\n❌ MOVE COUNT MISMATCH\n");
        fprintf(stderr, "FEN: %s\n", fen.c_str());
        fprintf(stderr, "Engine: %d  Stockfish: %d\n",
                engine_moves, sf_moves);

        board.print_board();
        for (int i = 0; i < engine_moves; ++i)
            fprintf(stderr, "%s\n", moves[i].move_to_string().c_str());

        return;
    }

    if (depth == 1)
        return;

    for (int i = 0; i < engine_moves; ++i) {
        StateInfo st;
        make_move(&board, moves[i], st);
        perft_debug_recursive(board, depth - 1, sf);
        undo_move(&board, moves[i], st);
    }
}

