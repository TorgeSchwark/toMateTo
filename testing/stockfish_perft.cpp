// stockfish_perft.cpp
//
// Sucht eine bereits vorhandene Stockfish-Binary im Ordner "stockfish_bin"
// (relativ zum aktuellen Arbeitsverzeichnis) und fuehrt den UCI-Befehl
// 'go perft <depth>' auf einer beliebigen FEN-Position aus.
//
// Kein Download, keine externen Abhaengigkeiten (kein libcurl noetig) —
// setzt voraus, dass stockfish_bin/ bereits eine lauffaehige Stockfish-
// Binary enthaelt.
//
// Build (Linux/macOS):
//   g++ -std=c++17 -O2 stockfish_perft.cpp -o stockfish_perft
//
// Nutzung:
//   ./stockfish_perft
//   ./stockfish_perft --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 5
//   ./stockfish_perft --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" --depth 4



#include "stockfish_perft.h"


// Standard: Schachbrett-Startposition
static const std::string START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static fs::path installDir() {
    return fs::current_path() / "testing" / "stockfish_bin";
}
static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

static bool endsWithCI(const std::string& s, const std::string& suffix) {
    std::string ls = toLower(s), lsuf = toLower(suffix);
    if (lsuf.size() > ls.size()) return false;
    return ls.compare(ls.size() - lsuf.size(), lsuf.size(), lsuf) == 0;
}

static bool isWindowsHost() {
    struct utsname uts;
    uname(&uts);
    std::string system = toLower(uts.sysname);
    return system.find("mingw") != std::string::npos ||
           system.find("windows") != std::string::npos;
}

// Sucht rekursiv im angegebenen Verzeichnis nach einer Datei, die mit
// "stockfish" beginnt (unter Windows zusaetzlich mit ".exe" endet, unter
// Linux/macOS keine Dateiendung hat).
static std::string findStockfishBinary(const fs::path& searchDir) {
    if (!fs::exists(searchDir)) return "";
    bool winHost = isWindowsHost();

    for (const auto& entry : fs::recursive_directory_iterator(searchDir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        std::string fl = toLower(filename);
        if (fl.rfind("stockfish", 0) != 0) continue; // muss mit "stockfish" beginnen

        if (winHost && endsWithCI(filename, ".exe")) {
            return entry.path().string();
        }
        if (!winHost && filename.find('.') == std::string::npos) {
            return entry.path().string();
        }
    }
    return "";
}

static std::string ensureStockfish() {
    fs::path dir = installDir();

    std::string existing = findStockfishBinary(dir);
    if (existing.empty()) {
        throw std::runtime_error(
            "Keine Stockfish-Binary in '" + dir.string() +
            "' gefunden. Bitte sicherstellen, dass der Ordner stockfish_bin "
            "eine lauffaehige Stockfish-Binary enthaelt.");
    }

    std::cout << "Stockfish gefunden: " << existing << "\n";

    if (!isWindowsHost()) {
        struct stat st{};
        stat(existing.c_str(), &st);
        chmod(existing.c_str(), st.st_mode | S_IXUSR | S_IXGRP | S_IXOTH);
    }

    return existing;
}

// ---------------------------------------------------------------------
// UCI-Kommunikation via fork/exec + Pipes (Aequivalent zu Popen.communicate)
// ---------------------------------------------------------------------

#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

static std::map<std::string, uint64_t> runPerft(
    const std::string& enginePath,
    const std::string& fen,
    int depth)
{
    std::vector<std::string> commands = {
        "uci",
        "isready",
        "position fen " + fen,
        "go perft " + std::to_string(depth)
    };

    std::ostringstream inputBuilder;

    for (const auto& c : commands) {
        inputBuilder << c << "\n";
    }

    inputBuilder << "quit\n";

    std::string inputStr = inputBuilder.str();


    int inPipe[2];   // Elternprozess -> Stockfish stdin
    int outPipe[2];  // Stockfish stdout -> Elternprozess

    if (pipe(inPipe) != 0 || pipe(outPipe) != 0) {
        throw std::runtime_error("pipe() fehlgeschlagen");
    }


    pid_t pid = fork();

    if (pid < 0) {
        throw std::runtime_error("fork() fehlgeschlagen");
    }


    if (pid == 0) {
        // ==========================================
        // KINDPROZESS = STOCKFISH
        // ==========================================

        dup2(inPipe[0], STDIN_FILENO);

        dup2(outPipe[1], STDOUT_FILENO);
        dup2(outPipe[1], STDERR_FILENO);

        close(inPipe[0]);
        close(inPipe[1]);

        close(outPipe[0]);
        close(outPipe[1]);

        execl(
            enginePath.c_str(),
            enginePath.c_str(),
            (char*)nullptr
        );

        // Wenn execl fehlschlägt
        _exit(127);
    }


    // ==========================================
    // ELTERNPROZESS
    // ==========================================

    close(inPipe[0]);
    close(outPipe[1]);


    // Befehle an Stockfish schicken
    ssize_t written = write(
        inPipe[1],
        inputStr.c_str(),
        inputStr.size()
    );

    (void)written;

    close(inPipe[1]);


    // ==========================================
    // STOCKFISH OUTPUT LESEN
    // ==========================================

    std::string output;

    char buf[4096];
    ssize_t n;

    while ((n = read(outPipe[0], buf, sizeof(buf))) > 0) {
        output.append(buf, n);
    }

    close(outPipe[0]);


    // Auf Stockfish warten
    int status = 0;

    waitpid(pid, &status, 0);


    // ==========================================
    // OUTPUT PARSEN
    // ==========================================

    std::map<std::string, uint64_t> result;

    std::istringstream stream(output);

    std::string line;

    while (std::getline(stream, line)) {

        /*
            Stockfish liefert z.B.:

            e2e4: 1200
            d2d4: 1300
            g1f3: 1400

            aber auch:

            Nodes searched: 123456
        */

        std::istringstream lineStream(line);

        std::string move;
        uint64_t nodes;


        if (lineStream >> move >> nodes) {

            // ":" vom Zug entfernen
            if (!move.empty() && move.back() == ':') {

                move.pop_back();

                result[move] = nodes;
            }
        }
    }


    return result;
}

static uint64_t sum_perft(
    const std::map<std::string, uint64_t>& perft)
{
    uint64_t total = 0;

    for (const auto& [move, count] : perft) {
        total += count;
    }

    return total;
}

static std::string debug_perft_position(
    chess_board* cb,
    const std::string& fen,
    int depth,
    const std::string& moveHistory)
{
    std::cout
        << "\n========================================\n"
        << "Debugging position\n"
        << "FEN: " << fen << "\n"
        << "Depth: " << depth << "\n";

    if (!moveHistory.empty()) {
        std::cout << "Moves: " << moveHistory << "\n";
    }

    std::cout
        << "========================================\n";


    // ============================================================
    // 1. STOCKFISH
    // ============================================================

    std::string enginePath = ensureStockfish();

    std::map<std::string, uint64_t> stockfishPerft =
        runPerft(
            enginePath,
            fen,
            depth
        );


    // ============================================================
    // 2. TOMATETO
    // ============================================================

    std::map<std::string, uint64_t> tomatoPerft =
        try_all_moves(
            cb,
            depth
        );


    // ============================================================
    // 3. AUSGABE
    // ============================================================

    uint64_t stockfishNodes =
        sum_perft(stockfishPerft);

    uint64_t tomatoNodes =
        sum_perft(tomatoPerft);


    std::cout
        << "Stockfish nodes: "
        << stockfishNodes
        << "\n";

    std::cout
        << "ToMateTo nodes:  "
        << tomatoNodes
        << "\n";


    // ============================================================
    // 4. ALLES OK?
    // ============================================================

    if (stockfishNodes == tomatoNodes) {

        std::cout
            << "MATCH at depth "
            << depth
            << "\n";

        return "";
    }


    // ============================================================
    // 5. MISMATCH
    // ============================================================

    std::cout
        << "MISMATCH at depth "
        << depth
        << "\n";


    // ============================================================
    // 6. ZUERST PRÜFEN:
    //
    // Gibt es einen Zug, den Stockfish hat,
    // ToMateTo aber NICHT?
    //
    // ODER umgekehrt?
    //
    // WICHTIG:
    // Das machen wir VOR dem Count-Vergleich.
    // ============================================================

    for (const auto& [move, stockfishCount] : stockfishPerft) {

        auto tomatoIt =
            tomatoPerft.find(move);


        if (tomatoIt == tomatoPerft.end()) {

            std::cout
                << "\n!!! MOVE MISSING IN TOMATETO !!!\n"
                << "Move: "
                << move
                << "\n"
                << "Stockfish: "
                << stockfishCount
                << "\n"
                << "ToMateTo: MISSING\n";


            return
                "Perft error found!\n"
                "FEN: " + fen + "\n"
                "Move history: " + moveHistory + "\n"
                "Missing move in ToMateTo: " + move + "\n"
                "Stockfish nodes: " +
                std::to_string(stockfishCount) +
                "\nToMateTo: MISSING\n";
        }
    }


    for (const auto& [move, tomatoCount] : tomatoPerft) {

        if (stockfishPerft.find(move) ==
            stockfishPerft.end()) {

            std::cout
                << "\n!!! MOVE MISSING IN STOCKFISH !!!\n"
                << "Move: "
                << move
                << "\n"
                << "ToMateTo: "
                << tomatoCount
                << "\n";


            return
                "Perft error found!\n"
                "FEN: " + fen + "\n"
                "Move history: " + moveHistory + "\n"
                "Missing move in Stockfish: " + move + "\n"
                "ToMateTo nodes: " +
                std::to_string(tomatoCount) +
                "\n";
        }
    }


    // ============================================================
    // 7. JETZT SIND BEIDE MAPS STRUKTURELL GLEICH
    //
    // Also muss der Fehler bei einem Zug COUNT liegen.
    //
    // Wir nehmen den ERSTEN Zug mit unterschiedlicher Anzahl.
    // ============================================================

    for (const auto& [move, stockfishCount] :
         stockfishPerft) {

        uint64_t tomatoCount =
            tomatoPerft.at(move);


        if (stockfishCount == tomatoCount) {

            // Dieser Zug ist korrekt.
            continue;
        }


        // ========================================================
        // ERSTER FEHLERHAFTER ZUG GEFUNDEN
        // ========================================================

        std::cout
            << "\n----------------------------------------\n"
            << "FIRST MISMATCH\n"
            << "Move: "
            << move
            << "\n"
            << "Stockfish: "
            << stockfishCount
            << "\n"
            << "ToMateTo: "
            << tomatoCount
            << "\n"
            << "----------------------------------------\n";


      

    

        // ========================================================
        // 8. WENN WIR NICHT MEHR TIEFER KÖNNEN
        // ========================================================

        if (depth <= 1) {

            std::cout
                << "\n!!! ERROR FOUND AT DEPTH 1 !!!\n";

            return
                "Perft error found!\n"
                "FEN: " + fen + "\n"
                "Move history: " + moveHistory + "\n"
                "Problem move: " + move + "\n"
                "Stockfish: " +
                std::to_string(stockfishCount) +
                "\nToMateTo: " +
                std::to_string(tomatoCount) +
                "\n";
        }


        // ========================================================
        // 9. DEN FEHLERHAFTEN ZUG AUF DEM BOARD SUCHEN
        // ========================================================

        Move moves[256];

        Move* end =
            find_all_moves(
                moves,
                cb
            );


        for (Move* m = moves; m != end; ++m) {

            if (m->move_to_string(cb->whites_turn) != move) {
                continue;
            }


            // ====================================================
            // 10. ZUG AUSFÜHREN
            // ====================================================

            StateInfo st;

            make_move(
                cb,
                *m,
                st
            );
            


            std::string newHistory =
                moveHistory.empty()
                    ? move
                    : moveHistory + " " + move;


            std::string newFen =
                board_to_fen(*cb);


            std::cout
                << "\nDescending into move: "
                << move
                << "\n"
                << "New FEN: "
                << newFen
                << "\n";


            // ====================================================
            // 11. REKURSIV EINE EBENE TIEFER
            // ====================================================

            std::string result =
                debug_perft_position(
                    cb,
                    newFen,
                    depth - 1,
                    newHistory
                );


            // ====================================================
            // 12. BOARD ZURÜCKSETZEN
            // ====================================================

            undo_move(
                cb,
                *m,
                st
            );


            return result;
        }


        // ========================================================
        // Sollte eigentlich nur passieren, wenn die Notation
        // zwischen Stockfish und ToMateTo unterschiedlich ist.
        // ========================================================

        return
            "Internal error: Stockfish move " +
            move +
            " could not be found in ToMateTo move list.";
    }


    return "Unknown perft mismatch.";
}
std::string find_perft_error(std::string fen)
{
    std::cout
        << "Searching Perft errors for FEN:\n  "
        << fen
        << "\n\n";

    try {

        chess_board cb;

        cb.setup_chess_board();

        setup_fen_position(
            cb,
            fen
        );


        // ========================================
        // Wir starten bei Depth 1
        // ========================================

        for (int depth = 1; depth < 8; depth++) {

            std::cout
                << "\n\n########################################\n"
                << "START DEPTH "
                << depth
                << "\n"
                << "########################################\n";


            std::string result =
                debug_perft_position(
                    &cb,
                    fen,
                    depth,
                    ""
                );


            // Kein Fehler auf dieser Tiefe
            if (result.empty()) {

                continue;
            }


            // Fehler gefunden
            std::cout
                << "\n\n"
                << "########################################\n"
                << "FINAL PERFT ERROR\n"
                << "########################################\n"
                << result
                << "\n";


            return result;
        }


        std::cout
            << "\nNo perft error found up to depth 7.\n";

    }
    catch (const std::exception& exc) {

        std::cerr
            << "Fehler: "
            << exc.what()
            << "\n";
    }


    return fen;
}

// int main(int argc, char** argv) {
//     std::string fen = START_FEN;
//     int depth = 5;

//     for (int i = 1; i < argc; ++i) {
//         std::string arg = argv[i];
//         if (arg == "--fen" && i + 1 < argc) {
//             fen = argv[++i];
//         } else if (arg == "--depth" && i + 1 < argc) {
//             depth = std::stoi(argv[++i]);
//         } else if (arg == "--help" || arg == "-h") {
//             std::cout << "Nutzung: " << argv[0]
//                       << " [--fen \"<FEN>\"] [--depth <n>]\n";
//             return 0;
//         }
//     }

//     try {
//         std::string enginePath = ensureStockfish();
//         std::cout << "\nFuehre 'go perft " << depth << "' aus fuer FEN:\n  " << fen
//                   << "\n\n";

//         std::string output = runPerft(enginePath, fen, depth);
//         std::cout << output;
//     } catch (const std::exception& exc) {
//         std::cerr << "Fehler: " << exc.what() << "\n";
//         return 1;
//     }

//     return 0;
// }

void test_position()
{
    chess_board board;

    setup_fen_position(
        board,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2"
    );

    std::cout << "Initial white king: "
              << __builtin_ctzll(board.white.king)
              << std::endl;

    std::cout << "Initial black king: "
              << __builtin_ctzll(board.black.king)
              << std::endl;


    // Züge nacheinander ausführen
    const std::vector<std::string> moves = {
        "a6b5",
        "f3g2",
        "a7a5",
        "e1g1",
        "e7c5",

    };

    for (const std::string& move_string : moves)
    {
        Move moves_buffer[256];
        Move* end = find_all_moves(moves_buffer, &board);

        bool found = false;

        for (Move* m = moves_buffer; m != end; ++m)
        {
            // Wichtig: move_to_string() vor make_move()
            // weil sich whites_turn nach make_move() ändert
            if (m->move_to_string(board.whites_turn) == move_string)
            {
                StateInfo st;

                std::cout << "\nExecuting: " << move_string << std::endl;

                make_move(&board, *m, st);

                found = true;
                break;
            }
        }

        if (!found)
        {
            std::cout << "ERROR: Move not found: "
                      << move_string << std::endl;
            break;
        }
        print_bitboard(board.white.king);
        // Nach make_move() wurde whites_turn bereits umgeschaltet.
        // Der gerade gezogene Spieler ist also die jeweils andere Farbe.
        std::cout << "White king square: ";

        if (board.white.king)
            std::cout << __builtin_ctzll(board.white.king);
        else
            std::cout << "MISSING";

        std::cout << std::endl;

        std::cout << "Black king square: ";

        if (board.black.king)
            std::cout << __builtin_ctzll(board.black.king);
        else
            std::cout << "MISSING";

        std::cout << std::endl;
    }
}