#pragma once
// Minimal persistent UCI engine process wrapper - spawns one Stockfish
// process and keeps it alive across many "position ...; go depth ..."
// round trips (spawning a fresh process per position, like
// testing/stockfish_perft.cpp's runPerft() does for occasional perft
// checks, would be far too slow for generating thousands of training
// positions).
//
// POSIX only (posix_spawn/pipe) - matches this project's existing Stockfish
// integration, which already assumes Linux/WSL.
//
// Uses posix_spawn(), not fork()+exec(): this class is constructed
// concurrently from many worker threads (full_cycle.cpp's --threads,
// strength_match.cpp's --threads), and plain fork() is unsafe to call from
// a multithreaded process except when the child does *nothing* but
// async-signal-safe calls before exec - found the hard way running
// strength_match.cpp with 8 threads, where several Stockfish subprocesses
// simply never came up (their pipe closed before sending "uciok"/a
// bestmove), while the identical setup at 1-4 threads never failed. glibc's
// posix_spawn is specifically implemented to be safe here (typically via
// clone/vfork with the child immediately exec'ing), which fork()+exec()
// hand-rolled in a threaded program is not guaranteed to be.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

class UciEngine {
public:
    explicit UciEngine(const std::string& path) {
        int in_pipe[2], out_pipe[2];
        if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) throw std::runtime_error("pipe() failed");

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, in_pipe[0]);
        posix_spawn_file_actions_addclose(&actions, in_pipe[1]);
        posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
        posix_spawn_file_actions_addclose(&actions, out_pipe[1]);

        char* argv[] = {const_cast<char*>(path.c_str()), nullptr};
        const int rc = posix_spawn(&pid_, path.c_str(), &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        if (rc != 0) throw std::runtime_error("posix_spawn() failed for " + path + ": " + std::strerror(rc));

        close(in_pipe[0]);
        close(out_pipe[1]);
        to_engine_ = fdopen(in_pipe[1], "w");
        from_engine_ = fdopen(out_pipe[0], "r");
        if (!to_engine_ || !from_engine_) throw std::runtime_error("fdopen() failed");

        send("uci");
        wait_for("uciok");
        // One process per worker thread (see full_cycle.cpp's --threads) is
        // how this project parallelizes Stockfish calls, not Stockfish's
        // own internal multithreading - force each instance to 1 thread so
        // N worker threads don't oversubscribe cores N times over.
        send("setoption name Threads value 1");
        send("isready");
        wait_for("readyok");
    }

    ~UciEngine() {
        if (to_engine_) {
            send("quit");
            fclose(to_engine_);
        }
        if (from_engine_) fclose(from_engine_);
        if (pid_ > 0) waitpid(pid_, nullptr, 0);
    }

    UciEngine(const UciEngine&) = delete;
    UciEngine& operator=(const UciEngine&) = delete;

    // Centipawn evaluation from the side-to-move's perspective (matches
    // nnue::train::TrainingSample::target_wdl's convention exactly - no
    // sign flip needed when turning this into a training target). Mate
    // scores are folded into a large-magnitude centipawn value so the
    // network at least learns "this is decisive", without the exact mate
    // distance.
    int evaluate(const std::string& fen, int depth) {
        send("position fen " + fen);
        send("go depth " + std::to_string(depth));

        int last_cp = 0;
        bool got_score = false;
        std::string line;
        while (read_line(line)) {
            if (auto pos = line.find("score cp "); pos != std::string::npos) {
                last_cp = std::stoi(line.substr(pos + 9));
                got_score = true;
            } else if (auto pos = line.find("score mate "); pos != std::string::npos) {
                int mate_in = std::stoi(line.substr(pos + 11));
                last_cp = mate_in > 0 ? (10000 - mate_in) : (-10000 - mate_in);
                got_score = true;
            }
            if (line.rfind("bestmove", 0) == 0) break;
        }
        if (!got_score) throw std::runtime_error("no 'score' line parsed for fen: " + fen);
        return last_cp;
    }

    // Generic "setoption name X value Y" - e.g. set_option("Hash", "16").
    void set_option(const std::string& name, const std::string& value) {
        send("setoption name " + name + " value " + value);
        send("isready");
        wait_for("readyok");
    }

    // Caps Stockfish at approximately this real-world Elo via its own
    // internal strength-limiting (not the older, coarser "Skill Level 0-20"
    // knob) - see strength_match.cpp, which bisects on this value to find
    // where the TestEngine's win rate crosses 50%.
    void set_limit_strength(int elo) {
        set_option("UCI_LimitStrength", "true");
        set_option("UCI_Elo", std::to_string(elo));
    }

    // Resets Stockfish's own hash table / move history between games (played
    // match games should not leak state into each other) - cheap, call once
    // per game before the first move.
    void new_game() {
        send("ucinewgame");
        send("isready");
        wait_for("readyok");
    }

    // Plays one move: returns its UCI notation (e.g. "e2e4", "e7e8q"), or
    // "(none)"/"0000" if Stockfish reports no legal move (checkmate/
    // stalemate reached from `fen` - the caller should already know this
    // from its own move generator, this is just here for symmetry/safety).
    std::string best_move(const std::string& fen, int movetime_ms) {
        send("position fen " + fen);
        send("go movetime " + std::to_string(movetime_ms));

        std::string line;
        while (read_line(line)) {
            if (line.rfind("bestmove", 0) == 0) {
                std::size_t start = 9; // strlen("bestmove ")
                std::size_t end = line.find_first_of(" \r\n", start);
                return line.substr(start, end == std::string::npos ? std::string::npos : end - start);
            }
        }
        throw std::runtime_error("engine closed its output before sending bestmove for fen: " + fen);
    }

private:
    void send(const std::string& cmd) {
        std::fputs((cmd + "\n").c_str(), to_engine_);
        std::fflush(to_engine_);
    }

    bool read_line(std::string& out) {
        char buf[4096];
        if (!std::fgets(buf, sizeof(buf), from_engine_)) return false;
        out.assign(buf);
        return true;
    }

    void wait_for(const std::string& token) {
        std::string line;
        while (read_line(line)) {
            if (line.find(token) != std::string::npos) return;
        }
        throw std::runtime_error("engine closed its output before sending: " + token);
    }

    pid_t pid_ = -1;
    FILE* to_engine_ = nullptr;
    FILE* from_engine_ = nullptr;
};
