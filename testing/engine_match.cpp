#include "engine_match.h"

static const std::string START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";


struct Stockfish
{
    pid_t pid;
    int in_fd;
    int out_fd;

    Stockfish(const std::string& path)
    {
        int inPipe[2];
        int outPipe[2];

        if (pipe(inPipe) || pipe(outPipe))
            throw std::runtime_error("pipe failed");

        pid = fork();

        if (pid < 0)
            throw std::runtime_error("fork failed");

        if (pid == 0)
        {
            dup2(inPipe[0], STDIN_FILENO);
            dup2(outPipe[1], STDOUT_FILENO);
            dup2(outPipe[1], STDERR_FILENO);

            close(inPipe[0]);
            close(inPipe[1]);
            close(outPipe[0]);
            close(outPipe[1]);

            execl(path.c_str(), path.c_str(), nullptr);
            _exit(127);
        }

        close(inPipe[0]);
        close(outPipe[1]);

        in_fd = inPipe[1];
        out_fd = outPipe[0];

        send("uci");
        wait_for("uciok");

        send("isready");
        wait_for("readyok");
    }

    ~Stockfish()
    {
        send("quit");

        close(in_fd);
        close(out_fd);

        waitpid(pid, nullptr, 0);
    }

    void send(const std::string& command)
    {
        std::string data = command + "\n";
        write(in_fd, data.c_str(), data.size());
    }

    std::string read_line()
    {
        std::string line;
        char c;

        while (true)
        {
            ssize_t n = read(out_fd, &c, 1);

            if (n <= 0)
                throw std::runtime_error("Stockfish pipe closed");

            if (c == '\n')
                break;

            if (c != '\r')
                line += c;
        }

        return line;
    }

    void wait_for(const std::string& wanted)
    {
        while (true)
        {
            std::string line = read_line();

            if (line == wanted)
                return;
        }
    }

    void set_elo(int elo)
    {
        send("setoption name UCI_LimitStrength value true");
        send("setoption name UCI_Elo value " + std::to_string(elo));

        send("isready");
        wait_for("readyok");
    }

    std::string best_move(
        const std::string& fen,
        double time_seconds)
    {
        send("position fen " + fen);

        int ms = static_cast<int>(time_seconds * 1000.0);

        send("go movetime " + std::to_string(ms));

        while (true)
        {
            std::string line = read_line();

            if (line.rfind("bestmove ", 0) == 0)
            {
                std::string move = line.substr(9);

                size_t space = move.find(' ');

                if (space != std::string::npos)
                    move = move.substr(0, space);

                return move;
            }
        }
    }
};


static bool make_uci_move(chess_board& board, const std::string& uci)
{
    MoveStacks moves;
    find_all_moves(&moves, &board);

    for (Move* m = moves.capture_moves; m != moves.capture_end; ++m)
    {
        if (m->move_to_string(board.whites_turn) == uci)
        {
            StateInfo st;
            make_move(&board, *m, st);
            return true;
        }
    }

    for (Move* m = moves.normal_moves; m != moves.normal_end; ++m)
    {
        if (m->move_to_string(board.whites_turn) == uci)
        {
            StateInfo st;
            make_move(&board, *m, st);
            return true;
        }
    }

    return false;
}


static bool has_legal_moves(chess_board& board)
{
    MoveStacks moves;
    find_all_moves(&moves, &board);

    return !moves.empty();
}


static int game_result(chess_board& board, bool engine_white)
{
    if (has_legal_moves(board))
        return 0;

    if (is_in_check(&board))
    {
        bool mated_white = board.whites_turn;

        if (mated_white == engine_white)
            return -1;
        else
            return 1;
    }

    return 2;
}


static int play_game(
    Stockfish& stockfish,
    double engine_time,
    int stockfish_elo,
    bool engine_white,
    int game_number)
{
    chess_board board;
    setup_fen_position(board, START_FEN);

    stockfish.set_elo(stockfish_elo);

    int ply = 0;

    while (true)
    {
        int result = game_result(board, engine_white);

        if (result != 0)
            return result;

        std::string fen = board_to_fen(board);

        std::string move;

        if (board.whites_turn == engine_white)
        {
            move = alpha_beta_tt_toMateTo(
                fen,
                engine_time
            );
        }
        else
        {
            move = stockfish.best_move(
                fen,
                engine_time
            );
        }

        if (move.empty())
            return (board.whites_turn == engine_white) ? -1 : 1;

        std::cout
            << "Game " << game_number
            << " move: "
            << move
            << " elo: " << stockfish_elo
            << "\n";

        if (!make_uci_move(board, move))
        {
            std::cerr
                << "Game " << game_number
                << " illegal move from engine: "
                << move
                << "\n"
                << "FEN: "
                << fen
                << "\n";

            return (board.whites_turn == engine_white) ? -1 : 1;
        }

        ++ply;

        if (ply >= 300)
            return 2;
    }
}


void run_engine_match_ST(
    double engine_time,
    int stockfish_elo)
{
    const std::string stockfish_path =
        "testing/stockfish_bin/stockfish/"
        "stockfish-ubuntu-x86-64-avx2";

    int total_games = 0;

    int engine_wins = 0;
    int stockfish_wins = 0;
    int draws = 0;

    while (true)
    {
        std::cout
            << "\n========================================\n"
            << "Stockfish Elo: "
            << stockfish_elo
            << "\n"
            << "========================================\n";

        int batch_engine_wins = 0;
        int batch_stockfish_wins = 0;
        int batch_draws = 0;

        for (int game_index = 0; game_index < 10; ++game_index)
        {
            int game_number = game_index + 1;

            bool engine_white = (game_index % 2 == 0);

            std::cout
                << "\nGame "
                << game_number
                << "/10  "
                << (engine_white
                        ? "ToMateTo white"
                        : "ToMateTo black")
                << "\n";

            Stockfish stockfish(stockfish_path);

            int result = play_game(
                stockfish,
                engine_time,
                stockfish_elo,
                engine_white,
                game_number
            );

            if (result == 1)
            {
                std::cout
                    << "Game "
                    << game_number
                    << " result: ToMateTo wins\n";

                ++engine_wins;
                ++batch_engine_wins;
            }
            else if (result == -1)
            {
                std::cout
                    << "Game "
                    << game_number
                    << " result: Stockfish wins\n";

                ++stockfish_wins;
                ++batch_stockfish_wins;
            }
            else
            {
                std::cout
                    << "Game "
                    << game_number
                    << " result: Draw\n";

                ++draws;
                ++batch_draws;
            }

            ++total_games;
        }

        double stockfish_points =
            batch_stockfish_wins +
            batch_draws * 0.5;

        std::cout
            << "\n----------------------------------------\n"
            << "Elo " << stockfish_elo << " results\n"
            << "----------------------------------------\n"
            << "ToMateTo wins:    "
            << batch_engine_wins
            << "\n"
            << "Stockfish wins:   "
            << batch_stockfish_wins
            << "\n"
            << "Draws:            "
            << batch_draws
            << "\n"
            << "Stockfish points: "
            << stockfish_points
            << "/10\n";

        if (stockfish_points > 7.0)
            break;

        stockfish_elo += 100;
    }

    std::cout
        << "\n\n========================================\n"
        << "FINAL RESULTS\n"
        << "========================================\n"
        << "Games:            "
        << total_games
        << "\n"
        << "ToMateTo wins:    "
        << engine_wins
        << "\n"
        << "Stockfish wins:   "
        << stockfish_wins
        << "\n"
        << "Draws:            "
        << draws
        << "\n"
        << "Stockfish points: "
        << stockfish_wins + draws * 0.5
        << "\n"
        << "========================================\n";
}


void run_engine_match(double engine_time, int stockfish_elo){

    const std::string stockfish_path =
        "testing/stockfish_bin/stockfish/"
        "stockfish-ubuntu-x86-64-avx2";

    const unsigned int hardware_cores =
        std::thread::hardware_concurrency();

    const unsigned int num_threads =
        std::min(hardware_cores == 0 ? 1u : hardware_cores, 10u);

    constexpr int GAMES_PER_ELO = 10;

    std::cout << "Available CPU cores: " << hardware_cores << "\n";
    std::cout << "Running " << num_threads << " games in parallel.\n";

    struct EloResult{
        int started = 0;
        int finished = 0;
        int engine_wins = 0;
        int stockfish_wins = 0;
        int draws = 0;
    };

    std::mutex mutex;
    std::condition_variable cv;

    std::map<int, EloResult> results;

    int next_elo = stockfish_elo;
    int game_number = 0;
    int total_games = 0;
    int engine_wins = 0;
    int stockfish_wins = 0;
    int draws = 0;

    bool stop = false;

    auto get_next_job = [&]() -> std::tuple<int, int, bool>{

        while (true){

            auto& elo_result = results[next_elo];

            if (elo_result.started < GAMES_PER_ELO){
                int game_index = elo_result.started++;
                int number = ++game_number;

                bool engine_white = (game_index % 2 == 0);

                return {next_elo, number, engine_white};
            }

            next_elo += 100;
            results[next_elo];
        }
    };

    auto worker = [&](){

        while (true){

            int elo;
            int number;
            bool engine_white;

            {
                std::unique_lock<std::mutex> lock(mutex);

                if (stop)
                    return;

                std::tie(elo, number, engine_white) = get_next_job();
            }

            std::cout
                << "Game " << number
                << " Elo " << elo
                << " "
                << (engine_white ? "ToMateTo white" : "ToMateTo black")
                << "\n";

            Stockfish stockfish(stockfish_path);

            int result = play_game(
                stockfish,
                engine_time,
                elo,
                engine_white,
                number
            );

            {
                std::lock_guard<std::mutex> lock(mutex);

                EloResult& elo_result = results[elo];

                ++elo_result.finished;
                ++total_games;

                if (result == 1){
                    ++elo_result.engine_wins;
                    ++engine_wins;
                }else if (result == -1){
                    ++elo_result.stockfish_wins;
                    ++stockfish_wins;
                }else{
                    ++elo_result.draws;
                    ++draws;
                }

                std::cout
                    << "Game " << number
                    << " result: "
                    << (result == 1 ? "ToMateTo wins" :
                        result == -1 ? "Stockfish wins" :
                        "Draw")
                    << " elo: " << elo
                    << "\n";

                if (elo_result.finished == GAMES_PER_ELO){

                    double stockfish_points =
                        elo_result.stockfish_wins +
                        elo_result.draws * 0.5;

                    std::cout
                        << "\n----------------------------------------\n"
                        << "Elo " << elo << " results\n"
                        << "----------------------------------------\n"
                        << "ToMateTo wins:    " << elo_result.engine_wins << "\n"
                        << "Stockfish wins:   " << elo_result.stockfish_wins << "\n"
                        << "Draws:            " << elo_result.draws << "\n"
                        << "Stockfish points: " << stockfish_points << "/10\n";

                    if (stockfish_points > 7.0)
                        stop = true;
                }
            }

            cv.notify_all();

            if (stop)
                return;
        }
    };

    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < num_threads; ++i)
        threads.emplace_back(worker);

    for (auto& thread : threads)
        thread.join();

    std::cout
        << "\n\n========================================\n"
        << "FINAL RESULTS\n"
        << "========================================\n"
        << "Games:            " << total_games << "\n"
        << "ToMateTo wins:    " << engine_wins << "\n"
        << "Stockfish wins:   " << stockfish_wins << "\n"
        << "Draws:            " << draws << "\n"
        << "Stockfish points: "
        << stockfish_wins + draws * 0.5
        << "\n"
        << "========================================\n";
}
