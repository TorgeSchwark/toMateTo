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
            fprintf(stderr, "%s\n", moves[i].move_to_string(board.whites_turn).c_str());

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


void perft_debug_depth2_undo(const std::string& fen) {

    chess_board board;
    board.setup_chess_board();
    setup_fen_position(board, fen);

    // ============================================================
    // Helper: collect all moves
    // ============================================================

    auto get_moves = [&](chess_board& b, Move* moves) -> int {
        Move* end = find_all_moves(moves, &b);
        return static_cast<int>(end - moves);
    };


    // ============================================================
    // Helper: check whether a move exists in a move list
    // ============================================================

    auto contains_move =
        [&](Move* moves,
            int count,
            const std::string& target,
            bool white_to_move) -> bool {

        for (int i = 0; i < count; ++i) {

            std::string move_str =
                moves[i].move_to_string(white_to_move);

            if (move_str == target)
                return true;
        }

        return false;
    };


    // ============================================================
    // Helper: compare two complete move lists
    //
    // Important:
    // We compare the actual moves, NOT just the number of moves.
    // ============================================================

    auto compare_move_lists =
        [&](Move* expected,
            int expected_count,
            Move* actual,
            int actual_count,
            bool white_to_move) -> bool {

        bool identical = true;

        // --------------------------------------------------------
        // Moves that disappeared
        // --------------------------------------------------------

        for (int i = 0; i < expected_count; ++i) {

            std::string expected_str =
                expected[i].move_to_string(white_to_move);

            if (!contains_move(
                    actual,
                    actual_count,
                    expected_str,
                    white_to_move)) {

                fprintf(stderr,
                        "    ❌ MOVE LOST: %s\n",
                        expected_str.c_str());

                identical = false;
            }
        }


        // --------------------------------------------------------
        // Moves that appeared
        // --------------------------------------------------------

        for (int i = 0; i < actual_count; ++i) {

            std::string actual_str =
                actual[i].move_to_string(white_to_move);

            if (!contains_move(
                    expected,
                    expected_count,
                    actual_str,
                    white_to_move)) {

                fprintf(stderr,
                        "    ❌ NEW / WRONG MOVE: %s\n",
                        actual_str.c_str());

                identical = false;
            }
        }

        return identical;
    };


    // ============================================================
    // Initial position
    // ============================================================

    bool initial_side = board.whites_turn;

    Move initial_moves[256];
    int initial_count =
        get_moves(board, initial_moves);


    fprintf(stderr,
            "\n"
            "==================================================\n"
            "        DEPTH 2 MAKE / UNDO DEBUG\n"
            "==================================================\n"
            "FEN:\n"
            "%s\n"
            "\n"
            "Side to move: %s\n"
            "Initial move count: %d\n"
            "==================================================\n\n",
            fen.c_str(),
            initial_side ? "WHITE" : "BLACK",
            initial_count
    );


    // ============================================================
    // Test every root move
    // ============================================================

    for (int i = 0; i < initial_count; ++i) {

        Move move1 = initial_moves[i];

        std::string move1_str =
            move1.move_to_string(board.whites_turn);


        fprintf(stderr,
                "[%d/%d] Testing: %s\n",
                i + 1,
                initial_count,
                move1_str.c_str());


        // --------------------------------------------------------
        // Make first move
        // --------------------------------------------------------

        StateInfo st1;

        make_move(&board, move1, st1);


        // --------------------------------------------------------
        // Position after move1
        // --------------------------------------------------------

        bool side_after_move1 =
            board.whites_turn;

        Move moves_after_1[256];

        int count_after_1 =
            get_moves(board, moves_after_1);


        // --------------------------------------------------------
        // Execute EVERY move2 and immediately undo it.
        //
        // We expect the position after move1 to be restored
        // after EVERY single move2.
        // --------------------------------------------------------

        for (int j = 0; j < count_after_1; ++j) {

            Move move2 = moves_after_1[j];

            std::string move2_str =
                move2.move_to_string(board.whites_turn);


            StateInfo st2;


            // ----------------------------------------------------
            // make move2
            // ----------------------------------------------------

            make_move(&board, move2, st2);


            // ----------------------------------------------------
            // undo move2
            // ----------------------------------------------------

            undo_move(&board, move2, st2);


            // ----------------------------------------------------
            // Check position after undo(move2)
            //
            // This MUST be identical to the position after move1.
            // ----------------------------------------------------

            Move check_moves[256];

            int check_count =
                get_moves(board, check_moves);


            if (check_count != count_after_1 ||
                !compare_move_lists(
                    moves_after_1,
                    count_after_1,
                    check_moves,
                    check_count,
                    side_after_move1)) {

                fprintf(stderr,
                        "\n"
                        "##################################################\n"
                        "❌ INNER UNDO FAILURE\n"
                        "##################################################\n"
                        "\n"
                        "Initial FEN:\n"
                        "%s\n"
                        "\n"
                        "Move 1:\n"
                        "    %s\n"
                        "\n"
                        "Move 2:\n"
                        "    %s\n"
                        "\n"
                        "Expected moves after undo(move2): %d\n"
                        "Actual moves:                     %d\n"
                        "\n"
                        "Position after move1 / undo(move2):\n",
                        fen.c_str(),
                        move1_str.c_str(),
                        move2_str.c_str(),
                        count_after_1,
                        check_count
                );

                board.print_board();

                fprintf(stderr,
                        "\n"
                        "The following operation corrupted the position:\n"
                        "\n"
                        "    make_move(%s)\n"
                        "    undo_move(%s)\n"
                        "\n"
                        "Expected the exact position after:\n"
                        "    %s\n"
                        "\n"
                        "##################################################\n",
                        move2_str.c_str(),
                        move2_str.c_str(),
                        move1_str.c_str()
                );

                return;
            }
        }


        // --------------------------------------------------------
        // Now undo move1
        // --------------------------------------------------------

        undo_move(&board, move1, st1);


        // --------------------------------------------------------
        // Check complete ORIGINAL move list.
        //
        // This is the important test for your current bug.
        //
        // It is NOT enough that the number is still 53.
        // We verify that g2g1b and every other original move
        // actually exists again.
        // --------------------------------------------------------

        Move restored_moves[256];

        int restored_count =
            get_moves(board, restored_moves);


        if (restored_count != initial_count ||
            !compare_move_lists(
                initial_moves,
                initial_count,
                restored_moves,
                restored_count,
                initial_side)) {

            fprintf(stderr,
                    "\n"
                    "##################################################\n"
                    "❌ OUTER UNDO FAILURE\n"
                    "##################################################\n"
                    "\n"
                    "Initial FEN:\n"
                    "%s\n"
                    "\n"
                    "Move that was made:\n"
                    "    %s\n"
                    "\n"
                    "Expected initial move count: %d\n"
                    "Actual move count:           %d\n"
                    "\n"
                    "The number can be equal while the actual move\n"
                    "list is different. Therefore the complete move\n"
                    "list was compared above.\n"
                    "\n"
                    "Restored board:\n",
                    fen.c_str(),
                    move1_str.c_str(),
                    initial_count,
                    restored_count
            );

            board.print_board();

            fprintf(stderr,
                    "\n"
                    "Operation tested:\n"
                    "    make_move(%s)\n"
                    "    ... all move2 make/undo tests ...\n"
                    "    undo_move(%s)\n"
                    "\n"
                    "The original position was NOT restored correctly.\n"
                    "##################################################\n",
                    move1_str.c_str(),
                    move1_str.c_str()
            );

            return;
        }
    }


    // ============================================================
    // SUCCESS
    // ============================================================

    fprintf(stderr,
            "\n"
            "==================================================\n"
            "✅ DEPTH 2 MAKE / UNDO TEST PASSED\n"
            "==================================================\n"
            "\n"
            "Initial moves: %d\n"
            "Tested every first move.\n"
            "Tested every second move.\n"
            "Every move2 was made and undone.\n"
            "Every restored position had the exact same move list.\n"
            "The original position had the exact same move list.\n"
            "\n"
            "==================================================\n",
            initial_count
    );
}


