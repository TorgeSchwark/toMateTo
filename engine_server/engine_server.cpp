// Kleiner Webserver für ToMateTo.
//
// Aufbau: Pro Verbindung wird ein eigener Prozess per fork() gestartet.
// Dadurch teilen sich parallele Spiele keinen Speicher (keine Probleme mit
// globalen Variablen / Transposition-Table in der Engine), und bis zu
// `max_engines` Berechnungen laufen echt parallel auf den 4 Kernen.
// Ein Semaphor (im Shared Memory) begrenzt die gleichzeitigen Suchen.

#include "engine_server.h"


namespace
{

sem_t* engine_slots = nullptr;


// ---------------------------------------------------------------- Helfer

std::string url_decode(const std::string& s)
{
    std::string out;

    for (size_t i = 0; i < s.size(); ++i)
    {
        if (s[i] == '%' && i + 2 < s.size())
        {
            out += static_cast<char>(
                std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16));
            i += 2;
        }
        else if (s[i] == '+')
            out += ' ';
        else
            out += s[i];
    }

    return out;
}


std::map<std::string, std::string> parse_query(const std::string& q)
{
    std::map<std::string, std::string> result;
    size_t pos = 0;

    while (pos < q.size())
    {
        size_t amp = q.find('&', pos);
        if (amp == std::string::npos)
            amp = q.size();

        std::string pair = q.substr(pos, amp - pos);
        size_t eq = pair.find('=');

        if (eq != std::string::npos)
            result[url_decode(pair.substr(0, eq))] =
                url_decode(pair.substr(eq + 1));

        pos = amp + 1;
    }

    return result;
}


bool fen_looks_valid(const std::string& fen)
{
    if (fen.empty() || fen.size() > 100)
        return false;

    for (char c : fen)
    {
        bool ok = std::isalnum(static_cast<unsigned char>(c)) ||
                  c == '/' || c == ' ' || c == '-';
        if (!ok)
            return false;
    }

    return true;
}


// ---------------------------------------------------------------- Schach

std::vector<std::string> legal_moves(chess_board& board)
{
    std::vector<std::string> result;

    MoveStacks moves;
    find_all_moves(&moves, &board);

    for (Move* m = moves.capture_moves; m != moves.capture_end; ++m)
        result.push_back(m->move_to_string(board.whites_turn));

    for (Move* m = moves.normal_moves; m != moves.normal_end; ++m)
        result.push_back(m->move_to_string(board.whites_turn));

    return result;
}


bool apply_uci_move(chess_board& board, const std::string& uci)
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


std::string position_json(chess_board& board, const std::string& prefix)
{
    std::vector<std::string> legal = legal_moves(board);
    bool check = is_in_check(&board);

    std::string status = "ongoing";
    if (legal.empty())
        status = check ? "checkmate" : "stalemate";

    std::string fen = board_to_fen(board);

    std::ostringstream json;
    json << "{" << prefix
         << "\"fen\":\"" << fen << "\","
         << "\"turn\":\"" << (board.whites_turn ? "w" : "b") << "\","
         << "\"check\":" << (check ? "true" : "false") << ","
         << "\"status\":\"" << status << "\","
         << "\"legal\":[";

    for (size_t i = 0; i < legal.size(); ++i)
    {
        if (i)
            json << ",";
        json << "\"" << legal[i] << "\"";
    }

    json << "]}";
    return json.str();
}


std::string error_json(const std::string& msg)
{
    return "{\"error\":\"" + msg + "\"}";
}


// ---------------------------------------------------------------- API

std::string handle_api(const std::string& route,
                       const std::map<std::string, std::string>& q)
{
    auto fen_it = q.find("fen");

    if (fen_it == q.end() || !fen_looks_valid(fen_it->second))
        return error_json("invalid fen");

    const std::string& fen = fen_it->second;

    chess_board board;
    setup_fen_position(board, fen);

    if (route == "/api/info")
        return position_json(board, "");

    if (route == "/api/move")
    {
        auto mv = q.find("move");

        if (mv == q.end() || !apply_uci_move(board, mv->second))
            return error_json("illegal move");

        return position_json(board, "\"move\":\"" + mv->second + "\",");
    }

    if (route == "/api/engine")
    {
        double seconds = 2.0;

        auto t = q.find("time");
        if (t != q.end())
            seconds = std::atof(t->second.c_str());

        if (seconds < 0.1)
            seconds = 0.1;
        if (seconds > 30.0)
            seconds = 30.0;

        if (legal_moves(board).empty())
            return error_json("game over");

        while (sem_wait(engine_slots) == -1 && errno == EINTR)
        {
        }

        Profiler::reset();
        std::string move = alpha_beta_tt_toMateTo(fen, seconds);
        Profiler::print();

        sem_post(engine_slots);

        if (move.empty() || !apply_uci_move(board, move))
            return error_json("engine returned no legal move");

        return position_json(board, "\"move\":\"" + move + "\",");
    }

    return error_json("unknown route");
}


// ---------------------------------------------------------------- HTTP

void send_response(int fd,
                   int status,
                   const std::string& type,
                   const std::string& body)
{
    const char* text = status == 200 ? "OK"
                     : status == 404 ? "Not Found"
                                     : "Bad Request";

    std::ostringstream head;
    head << "HTTP/1.1 " << status << " " << text << "\r\n"
         << "Content-Type: " << type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Cache-Control: no-store\r\n"
         << "Connection: close\r\n\r\n";

    std::string data = head.str() + body;

    size_t sent = 0;
    while (sent < data.size())
    {
        ssize_t n = write(fd, data.data() + sent, data.size() - sent);
        if (n <= 0)
            break;
        sent += static_cast<size_t>(n);
    }
}


void handle_client(int fd, const std::string& web_dir)
{
    std::string request;
    char buf[2048];

    while (request.find("\r\n\r\n") == std::string::npos &&
           request.size() < 16384)
    {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n <= 0)
            return;
        request.append(buf, static_cast<size_t>(n));
    }

    std::istringstream line(request.substr(0, request.find("\r\n")));
    std::string method, target, version;
    line >> method >> target >> version;

    if (method != "GET")
    {
        send_response(fd, 400, "text/plain", "only GET supported");
        return;
    }

    std::string route = target;
    std::string query;

    size_t qm = target.find('?');
    if (qm != std::string::npos)
    {
        route = target.substr(0, qm);
        query = target.substr(qm + 1);
    }

    if (route == "/" || route == "/index.html")
    {
        std::ifstream file(web_dir + "/index.html", std::ios::binary);

        if (!file)
        {
            send_response(fd, 404, "text/plain",
                          "index.html not found in " + web_dir);
            return;
        }

        std::stringstream content;
        content << file.rdbuf();
        send_response(fd, 200, "text/html; charset=utf-8", content.str());
        return;
    }

    if (route.rfind("/api/", 0) == 0)
    {
        send_response(fd, 200, "application/json",
                      handle_api(route, parse_query(query)));
        return;
    }

    send_response(fd, 404, "text/plain", "not found");
}

}


// ---------------------------------------------------------------- Start

void run_web_server(int port, const std::string& web_dir, int max_engines)
{
    signal(SIGCHLD, SIG_IGN);
    signal(SIGPIPE, SIG_IGN);

    engine_slots = static_cast<sem_t*>(
        mmap(nullptr, sizeof(sem_t), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));

    if (engine_slots == MAP_FAILED ||
        sem_init(engine_slots, 1, static_cast<unsigned>(max_engines)) != 0)
        throw std::runtime_error("semaphore init failed");

    int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0)
        throw std::runtime_error("socket failed");

    int yes = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error("bind failed (Port belegt?)");

    if (listen(server, 32) < 0)
        throw std::runtime_error("listen failed");

    std::cout << "ToMateTo laeuft auf http://0.0.0.0:" << port
              << "  (" << max_engines << " Engine-Slots)\n";

    while (true)
    {
        int client = accept(server, nullptr, nullptr);

        if (client < 0)
        {
            if (errno == EINTR)
                continue;
            break;
        }

        pid_t pid = fork();

        if (pid == 0)
        {
            close(server);
            alarm(120);
            handle_client(client, web_dir);
            close(client);
            _exit(0);
        }

        close(client);
    }

    close(server);
}
