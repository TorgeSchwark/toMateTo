#ifndef STOCKFISH_PERFT
#define STOCKFISH_PERFT

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdint>
#include <map>

#include "toMateTo_engine/move_generation/chess_board.h"

namespace fs = std::filesystem;

static fs::path installDir();

static std::string toLower(std::string s);

static bool endsWithCI(const std::string& s, const std::string& suffix);

static bool isWindowsHost();

static std::string findStockfishBinary(const fs::path& searchDir);

static std::string ensureStockfish();


static std::string debug_perft_position(
    chess_board* cb,
    const std::string& fen,
    int depth,
    const std::string& moveHistory);

static std::map<std::string, uint64_t> runPerft(
    const std::string& enginePath,
    const std::string& fen,
    int depth);


uint64_t get_node_count(const std::string& output);

std::string find_perft_error(std::string fen);

void test_position();

#endif 

