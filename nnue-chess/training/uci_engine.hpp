#pragma once
// Minimal persistent UCI engine process wrapper - spawns one Stockfish
// process and keeps it alive across many "position ...; go depth ..."
// round trips (spawning a fresh process per position, like
// testing/stockfish_perft.cpp's runPerft() does for occasional perft
// checks, would be far too slow for generating thousands of training
// positions).
//
// POSIX only (fork/pipe/exec) - matches this project's existing Stockfish
// integration, which already assumes Linux/WSL.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

class UciEngine {
public:
    explicit UciEngine(const std::string& path) {
        int in_pipe[2], out_pipe[2];
        if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) throw std::runtime_error("pipe() failed");

        pid_ = fork();
        if (pid_ < 0) throw std::runtime_error("fork() failed");

        if (pid_ == 0) {
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe[0]);
            close(in_pipe[1]);
            close(out_pipe[0]);
            close(out_pipe[1]);
            execl(path.c_str(), path.c_str(), (char*)nullptr);
            _exit(127); // execl only returns on failure
        }

        close(in_pipe[0]);
        close(out_pipe[1]);
        to_engine_ = fdopen(in_pipe[1], "w");
        from_engine_ = fdopen(out_pipe[0], "r");
        if (!to_engine_ || !from_engine_) throw std::runtime_error("fdopen() failed");

        send("uci");
        wait_for("uciok");
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
