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
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);

        close(in_pipe[1]);
        close(out_pipe[0]);

        execl(path, path, nullptr);
        _exit(1);
    }

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


void perft_debug_recursive(chess_board& board, int depth, Process& sf) {
    MoveStacks moves;
    find_all_moves(&moves, &board);

    int engine_moves = moves.capture_size() + moves.normal_size();

    std::string fen = board_to_fen(board);
    int sf_moves = stockfish_move_count(sf, fen);

    if (engine_moves != sf_moves) {
        fprintf(stderr, "\n❌ MOVE COUNT MISMATCH\n");
        fprintf(stderr, "FEN: %s\n", fen.c_str());
        fprintf(stderr, "Engine: %d  Stockfish: %d\n", engine_moves, sf_moves);

        board.print_board();

        for (Move* m = moves.capture_moves; m != moves.capture_end; ++m)
            fprintf(stderr, "%s\n", m->move_to_string(board.whites_turn).c_str());

        for (Move* m = moves.normal_moves; m != moves.normal_end; ++m)
            fprintf(stderr, "%s\n", m->move_to_string(board.whites_turn).c_str());

        return;
    }

    if (depth == 1)
        return;

    for (Move* m = moves.capture_moves; m != moves.capture_end; ++m) {
        StateInfo st;
        make_move(&board, *m, st);
        perft_debug_recursive(board, depth - 1, sf);
        undo_move(&board, *m, st);
    }

    for (Move* m = moves.normal_moves; m != moves.normal_end; ++m) {
        StateInfo st;
        make_move(&board, *m, st);
        perft_debug_recursive(board, depth - 1, sf);
        undo_move(&board, *m, st);
    }
}


void perft_debug_depth2_undo(const std::string& fen) {

    chess_board board;
    board.setup_chess_board();
    setup_fen_position(board, fen);

    auto get_moves = [&](chess_board& b, MoveStacks& moves) {
        find_all_moves(&moves, &b);
    };

    auto move_count = [&](MoveStacks& moves) {
        return moves.capture_size() + moves.normal_size();
    };

    auto contains_move = [&](MoveStacks& moves, const std::string& target, bool white_to_move) -> bool {
        for (Move* m = moves.capture_moves; m != moves.capture_end; ++m) {
            if (m->move_to_string(white_to_move) == target)
                return true;
        }

        for (Move* m = moves.normal_moves; m != moves.normal_end; ++m) {
            if (m->move_to_string(white_to_move) == target)
                return true;
        }

        return false;
    };

    auto compare_move_lists = [&](MoveStacks& expected, MoveStacks& actual, bool white_to_move) -> bool {
        bool identical = true;

        for (Move* m = expected.capture_moves; m != expected.capture_end; ++m) {
            std::string expected_str = m->move_to_string(white_to_move);

            if (!contains_move(actual, expected_str, white_to_move)) {
                fprintf(stderr, "    ❌ MOVE LOST: %s\n", expected_str.c_str());
                identical = false;
            }
        }

        for (Move* m = expected.normal_moves; m != expected.normal_end; ++m) {
            std::string expected_str = m->move_to_string(white_to_move);

            if (!contains_move(actual, expected_str, white_to_move)) {
                fprintf(stderr, "    ❌ MOVE LOST: %s\n", expected_str.c_str());
                identical = false;
            }
        }

        for (Move* m = actual.capture_moves; m != actual.capture_end; ++m) {
            std::string actual_str = m->move_to_string(white_to_move);

            if (!contains_move(expected, actual_str, white_to_move)) {
                fprintf(stderr, "    ❌ NEW / WRONG MOVE: %s\n", actual_str.c_str());
                identical = false;
            }
        }

        for (Move* m = actual.normal_moves; m != actual.normal_end; ++m) {
            std::string actual_str = m->move_to_string(white_to_move);

            if (!contains_move(expected, actual_str, white_to_move)) {
                fprintf(stderr, "    ❌ NEW / WRONG MOVE: %s\n", actual_str.c_str());
                identical = false;
            }
        }

        return identical;
    };

    bool initial_side = board.whites_turn;

    MoveStacks initial_moves;
    get_moves(board, initial_moves);

    int initial_count = move_count(initial_moves);

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

    int move1_index = 0;

    for (Move* m1 = initial_moves.capture_moves; m1 != initial_moves.capture_end; ++m1) {

        Move move1 = *m1;

        std::string move1_str =
            move1.move_to_string(board.whites_turn);

        fprintf(stderr,
                "[%d/%d] Testing: %s\n",
                ++move1_index,
                initial_count,
                move1_str.c_str());

        StateInfo st1;

        make_move(&board, move1, st1);

        bool side_after_move1 = board.whites_turn;

        MoveStacks moves_after_1;
        get_moves(board, moves_after_1);

        int count_after_1 = move_count(moves_after_1);

        int move2_index = 0;

        for (Move* m2 = moves_after_1.capture_moves; m2 != moves_after_1.capture_end; ++m2) {

            Move move2 = *m2;

            std::string move2_str =
                move2.move_to_string(board.whites_turn);

            StateInfo st2;

            make_move(&board, move2, st2);
            undo_move(&board, move2, st2);

            MoveStacks check_moves;
            get_moves(board, check_moves);

            int check_count = move_count(check_moves);

            if (check_count != count_after_1 ||
                !compare_move_lists(
                    moves_after_1,
                    check_moves,
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

        for (Move* m2 = moves_after_1.normal_moves; m2 != moves_after_1.normal_end; ++m2) {

            Move move2 = *m2;

            std::string move2_str =
                move2.move_to_string(board.whites_turn);

            StateInfo st2;

            make_move(&board, move2, st2);
            undo_move(&board, move2, st2);

            MoveStacks check_moves;
            get_moves(board, check_moves);

            int check_count = move_count(check_moves);

            if (check_count != count_after_1 ||
                !compare_move_lists(
                    moves_after_1,
                    check_moves,
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

        undo_move(&board, move1, st1);

        MoveStacks restored_moves;
        get_moves(board, restored_moves);

        int restored_count = move_count(restored_moves);

        if (restored_count != initial_count ||
            !compare_move_lists(
                initial_moves,
                restored_moves,
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

    for (Move* m1 = initial_moves.normal_moves; m1 != initial_moves.normal_end; ++m1) {

        Move move1 = *m1;

        std::string move1_str =
            move1.move_to_string(board.whites_turn);

        fprintf(stderr,
                "[%d/%d] Testing: %s\n",
                ++move1_index,
                initial_count,
                move1_str.c_str());

        StateInfo st1;

        make_move(&board, move1, st1);

        bool side_after_move1 = board.whites_turn;

        MoveStacks moves_after_1;
        get_moves(board, moves_after_1);

        int count_after_1 = move_count(moves_after_1);

        for (Move* m2 = moves_after_1.capture_moves; m2 != moves_after_1.capture_end; ++m2) {

            Move move2 = *m2;

            std::string move2_str =
                move2.move_to_string(board.whites_turn);

            StateInfo st2;

            make_move(&board, move2, st2);
            undo_move(&board, move2, st2);

            MoveStacks check_moves;
            get_moves(board, check_moves);

            int check_count = move_count(check_moves);

            if (check_count != count_after_1 ||
                !compare_move_lists(
                    moves_after_1,
                    check_moves,
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

        for (Move* m2 = moves_after_1.normal_moves; m2 != moves_after_1.normal_end; ++m2) {

            Move move2 = *m2;

            std::string move2_str =
                move2.move_to_string(board.whites_turn);

            StateInfo st2;

            make_move(&board, move2, st2);
            undo_move(&board, move2, st2);

            MoveStacks check_moves;
            get_moves(board, check_moves);

            int check_count = move_count(check_moves);

            if (check_count != count_after_1 ||
                !compare_move_lists(
                    moves_after_1,
                    check_moves,
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

        undo_move(&board, move1, st1);

        MoveStacks restored_moves;
        get_moves(board, restored_moves);

        int restored_count = move_count(restored_moves);

        if (restored_count != initial_count ||
            !compare_move_lists(
                initial_moves,
                restored_moves,
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
