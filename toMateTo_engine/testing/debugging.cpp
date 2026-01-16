#include "debugging.h"



struct Process {
    int in_fd;   // write to child
    int out_fd;  // read from child
    pid_t pid;
};

int stockfish_move_count(Process& sf, const std::string& fen) {
    char buf[512];

    // Position setzen
    std::string cmd = "position fen " + fen + "\n";
    write(sf.in_fd, cmd.c_str(), cmd.size());

    write(sf.in_fd, "isready\n", 8);
    while (read(sf.out_fd, buf, sizeof(buf)) > 0) {
        if (strstr(buf, "readyok")) break;
    }

    // Perft depth 1
    write(sf.in_fd, "go perft 1\n", 11);

    int nodes = -1;
    while (read(sf.out_fd, buf, sizeof(buf)) > 0) {
        if (strstr(buf, "Nodes searched")) {
            sscanf(buf, "info string Nodes searched: %d", &nodes);
            break;
        }
    }

    return nodes;
}


Process start_stockfish(const char* path) {
    int in_pipe[2];
    int out_pipe[2];
    pipe(in_pipe);
    pipe(out_pipe);

    pid_t pid = fork();
    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[1]);
        close(out_pipe[0]);
        execl(path, path, nullptr);
        _exit(1);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);

    return { in_pipe[1], out_pipe[0], pid };
}


void perft_debugging(const std::string& fen, int depth) {
    printf("so far");

    chess_board chess_board;
    chess_board.setup_chess_board();
    setup_fen_position(chess_board, fen);

    Move moves[256];
    Move* end = find_all_moves(moves, &chess_board);
    int num_moves = end - moves;

    // Stockfish starten (einmal pro Aufruf ok für Debugging)
    printf("so far");

    auto sf = start_stockfish("./toMateTo/testing/stockfish-ubuntu-x86-64-avx2");
    write(sf.in_fd, "uci\n", 4);

    char buf[256];
    while (read(sf.out_fd, buf, sizeof(buf)) > 0) {
        if (strstr(buf, "uciok")) break;
    }

    int sf_moves = stockfish_move_count(sf, fen);

    // 🔴 Abweichung → ausgeben und abbrechen
    if (num_moves != sf_moves) {
        printf("\n❌ MOVE COUNT MISMATCH\n");
        printf("FEN: %s\n", fen.c_str());
        printf("Engine moves: %d\n", num_moves);
        printf("Stockfish:   %d\n\n", sf_moves);

        chess_board.print_board();

        printf("\nMoves found by engine:\n");
        for (int i = 0; i < num_moves; ++i) {
            printf("%s\n", moves[i].move_to_string().c_str());
        }

        return;
    }

    // ✅ Tiefe 0 erreicht
    if (depth <= 1)
        return;

    // 🔁 Rekursiv: nach jedem Zug tiefer
    for (int i = 0; i < num_moves; ++i) {
        Move m = moves[i];
        StateInfo st;

        make_move(&chess_board, m, st);

        std::string next_fen = board_to_fen(chess_board);
        perft_debugging(next_fen, depth - 1);

        undo_move(&chess_board, m, st);
    }
}
