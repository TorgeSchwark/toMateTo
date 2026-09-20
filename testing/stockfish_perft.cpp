#include "stockfish_perft.h"

static const std::string START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static fs::path installDir(){
    return fs::current_path() / "testing" / "stockfish_bin";
}

static std::string toLower(std::string s){
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

static bool endsWithCI(const std::string& s, const std::string& suffix){
    std::string ls = toLower(s), lsuf = toLower(suffix);
    return lsuf.size() <= ls.size() &&
           ls.compare(ls.size() - lsuf.size(), lsuf.size(), lsuf) == 0;
}

static bool isWindowsHost(){
    struct utsname uts;
    uname(&uts);
    std::string system = toLower(uts.sysname);
    return system.find("mingw") != std::string::npos ||
           system.find("windows") != std::string::npos;
}

static std::string findStockfishBinary(const fs::path& searchDir){
    if(!fs::exists(searchDir)) return "";

    bool winHost = isWindowsHost();

    for(const auto& entry : fs::recursive_directory_iterator(searchDir)){
        if(!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        std::string lower = toLower(filename);

        if(lower.rfind("stockfish", 0) != 0) continue;

        if(winHost && endsWithCI(filename, ".exe"))
            return entry.path().string();

        if(!winHost && filename.find('.') == std::string::npos)
            return entry.path().string();
    }

    return "";
}

static std::string ensureStockfish(){
    fs::path dir = installDir();
    std::string existing = findStockfishBinary(dir);

    if(existing.empty())
        throw std::runtime_error(
            "Keine Stockfish-Binary in '" + dir.string() +
            "' gefunden. Bitte sicherstellen, dass der Ordner stockfish_bin "
            "eine lauffaehige Stockfish-Binary enthaelt.");

    std::cout << "Stockfish gefunden: " << existing << "\n";

    if(!isWindowsHost()){
        struct stat st{};
        stat(existing.c_str(), &st);
        chmod(existing.c_str(), st.st_mode | S_IXUSR | S_IXGRP | S_IXOTH);
    }

    return existing;
}

static std::map<std::string, uint64_t> runPerft(
    const std::string& enginePath,
    const std::string& fen,
    int depth)
{
    std::string input =
        "uci\n"
        "isready\n"
        "position fen " + fen + "\n"
        "go perft " + std::to_string(depth) + "\n"
        "quit\n";

    int inPipe[2], outPipe[2];

    if(pipe(inPipe) != 0 || pipe(outPipe) != 0)
        throw std::runtime_error("pipe() fehlgeschlagen");

    pid_t pid = fork();

    if(pid < 0)
        throw std::runtime_error("fork() fehlgeschlagen");

    if(pid == 0){
        dup2(inPipe[0], STDIN_FILENO);
        dup2(outPipe[1], STDOUT_FILENO);
        dup2(outPipe[1], STDERR_FILENO);

        close(inPipe[0]);
        close(inPipe[1]);
        close(outPipe[0]);
        close(outPipe[1]);

        execl(enginePath.c_str(), enginePath.c_str(), (char*)nullptr);
        _exit(127);
    }

    close(inPipe[0]);
    close(outPipe[1]);

    write(inPipe[1], input.c_str(), input.size());
    close(inPipe[1]);

    std::string output;
    char buf[4096];
    ssize_t n;

    while((n = read(outPipe[0], buf, sizeof(buf))) > 0)
        output.append(buf, n);

    close(outPipe[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    std::map<std::string, uint64_t> result;
    std::istringstream stream(output);
    std::string line;

    while(std::getline(stream, line)){
        std::istringstream lineStream(line);
        std::string move;
        uint64_t nodes;

        if(lineStream >> move >> nodes && !move.empty() && move.back() == ':'){
            move.pop_back();
            result[move] = nodes;
        }
    }

    return result;
}

static uint64_t sum_perft(const std::map<std::string, uint64_t>& perft){
    uint64_t total = 0;

    for(const auto& [move, count] : perft)
        total += count;

    return total;
}

static std::string debug_perft_position(
    chess_board* cb,
    const std::string& fen,
    int depth,
    const std::string& moveHistory)
{
    std::cout << "\n========================================\n"
              << "Debugging position\n"
              << "FEN: " << fen << "\n"
              << "Depth: " << depth << "\n";

    if(!moveHistory.empty())
        std::cout << "Moves: " << moveHistory << "\n";

    std::cout << "========================================\n";

    std::string enginePath = ensureStockfish();

    std::map<std::string, uint64_t> stockfishPerft =
        runPerft(enginePath, fen, depth);

    std::map<std::string, uint64_t> tomatoPerft =
        try_all_moves(cb, depth);

    uint64_t stockfishNodes = sum_perft(stockfishPerft);
    uint64_t tomatoNodes = sum_perft(tomatoPerft);

    std::cout << "Stockfish nodes: " << stockfishNodes << "\n"
              << "ToMateTo nodes:  " << tomatoNodes << "\n";

    if(stockfishNodes == tomatoNodes){
        std::cout << "MATCH at depth " << depth << "\n";
        return "";
    }

    std::cout << "MISMATCH at depth " << depth << "\n";

    for(const auto& [move, stockfishCount] : stockfishPerft){
        auto tomatoIt = tomatoPerft.find(move);

        if(tomatoIt == tomatoPerft.end()){
            std::cout << "\n!!! MOVE MISSING IN TOMATETO !!!\n"
                      << "Move: " << move << "\n"
                      << "Stockfish: " << stockfishCount << "\n"
                      << "ToMateTo: MISSING\n";

            return "Perft error found!\n"
                   "FEN: " + fen + "\n"
                   "Move history: " + moveHistory + "\n"
                   "Missing move in ToMateTo: " + move + "\n"
                   "Stockfish nodes: " + std::to_string(stockfishCount) +
                   "\nToMateTo: MISSING\n";
        }
    }

    for(const auto& [move, tomatoCount] : tomatoPerft){
        if(stockfishPerft.find(move) == stockfishPerft.end()){
            std::cout << "\n!!! MOVE MISSING IN STOCKFISH !!!\n"
                      << "Move: " << move << "\n"
                      << "ToMateTo: " << tomatoCount << "\n";

            return "Perft error found!\n"
                   "FEN: " + fen + "\n"
                   "Move history: " + moveHistory + "\n"
                   "Missing move in Stockfish: " + move + "\n"
                   "ToMateTo nodes: " + std::to_string(tomatoCount) + "\n";
        }
    }

    for(const auto& [move, stockfishCount] : stockfishPerft){
        uint64_t tomatoCount = tomatoPerft.at(move);

        if(stockfishCount == tomatoCount)
            continue;

        std::cout << "\n----------------------------------------\n"
                  << "FIRST MISMATCH\n"
                  << "Move: " << move << "\n"
                  << "Stockfish: " << stockfishCount << "\n"
                  << "ToMateTo: " << tomatoCount << "\n"
                  << "----------------------------------------\n";

        if(depth <= 1){
            std::cout << "\n!!! ERROR FOUND AT DEPTH 1 !!!\n";

            return "Perft error found!\n"
                   "FEN: " + fen + "\n"
                   "Move history: " + moveHistory + "\n"
                   "Problem move: " + move + "\n"
                   "Stockfish: " + std::to_string(stockfishCount) +
                   "\nToMateTo: " + std::to_string(tomatoCount) + "\n";
        }

        MoveStacks moves;
        find_all_moves(&moves, cb);

        for(Move* m = moves.capture_moves; m != moves.capture_end; ++m){
            if(m->move_to_string(cb->whites_turn) != move)
                continue;

            StateInfo st;
            make_move(cb, *m, st);

            std::string newHistory =
                moveHistory.empty() ? move : moveHistory + " " + move;
            std::string newFen = board_to_fen(*cb);

            std::cout << "\nDescending into move: " << move
                      << "\nNew FEN: " << newFen << "\n";

            std::string result =
                debug_perft_position(cb, newFen, depth - 1, newHistory);

            undo_move(cb, *m, st);
            return result;
        }

        for(Move* m = moves.normal_moves; m != moves.normal_end; ++m){
            if(m->move_to_string(cb->whites_turn) != move)
                continue;

            StateInfo st;
            make_move(cb, *m, st);

            std::string newHistory =
                moveHistory.empty() ? move : moveHistory + " " + move;
            std::string newFen = board_to_fen(*cb);

            std::cout << "\nDescending into move: " << move
                      << "\nNew FEN: " << newFen << "\n";

            std::string result =
                debug_perft_position(cb, newFen, depth - 1, newHistory);

            undo_move(cb, *m, st);
            return result;
        }

        return "Internal error: Stockfish move " + move +
               " could not be found in ToMateTo move list.";
    }

    return "Unknown perft mismatch.";
}

std::string find_perft_error(std::string fen){
    std::cout << "Searching Perft errors for FEN:\n  "
              << fen << "\n\n";

    try{
        chess_board cb;
        cb.setup_chess_board();
        setup_fen_position(cb, fen);

        for(int depth = 1; depth < 7; ++depth){
            std::cout << "\n\n########################################\n"
                      << "START DEPTH " << depth << "\n"
                      << "########################################\n";

            std::string result =
                debug_perft_position(&cb, fen, depth, "");

            if(!result.empty()){
                std::cout << "\n\n########################################\n"
                          << "FINAL PERFT ERROR\n"
                          << "########################################\n"
                          << result << "\n";
                return result;
            }
        }

        std::cout << "\nNo perft error found up to depth 7.\n";
    }
    catch(const std::exception& exc){
        std::cerr << "Fehler: " << exc.what() << "\n";
    }

    return fen;
}

void test_position(){
    chess_board board;

    setup_fen_position(
        board,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2"
    );

    std::cout << "Initial white king: "
              << __builtin_ctzll(board.white.king) << "\n"
              << "Initial black king: "
              << __builtin_ctzll(board.black.king) << "\n";

    const std::vector<std::string> move_list = {
        "a6b5", "f3g2", "a7a5", "e1g1", "e7c5"
    };

    for(const std::string& move_string : move_list){
        MoveStacks moves;
        find_all_moves(&moves, &board);

        Move* found_move = nullptr;

        for(Move* m = moves.capture_moves; m != moves.capture_end; ++m){
            if(m->move_to_string(board.whites_turn) == move_string){
                found_move = m;
                break;
            }
        }

        if(!found_move){
            for(Move* m = moves.normal_moves; m != moves.normal_end; ++m){
                if(m->move_to_string(board.whites_turn) == move_string){
                    found_move = m;
                    break;
                }
            }
        }

        if(!found_move){
            std::cout << "ERROR: Move not found: "
                      << move_string << "\n";
            break;
        }

        StateInfo st;

        std::cout << "\nExecuting: "
                  << move_string << "\n";

        make_move(&board, *found_move, st);

        print_bitboard(board.white.king);

        std::cout << "White king square: ";
        if(board.white.king)
            std::cout << __builtin_ctzll(board.white.king);
        else
            std::cout << "MISSING";

        std::cout << "\nBlack king square: ";
        if(board.black.king)
            std::cout << __builtin_ctzll(board.black.king);
        else
            std::cout << "MISSING";

        std::cout << "\n";
    }
}
