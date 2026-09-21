#include "chess_board.h"



// ============================================================
//  Move-Ausgabe
// ============================================================






// ============================================================
//  Bauern
// ============================================================
static inline void find_different_pawn_moves(Bitboard pawns, Bitboard empty, one_side* player, one_side* enemy,
                                             chess_board* cb, Bitboard* results)
{
    const bool is_white = cb->whites_turn;

    results[PUSH1] = BB_SHIFT(pawns, FORWARD, is_white) & empty;
    Bitboard p2 = BB_SHIFT(pawns & PAWN_ROW[is_white], FORWARD, is_white) & empty;
    results[PUSH2] = BB_SHIFT(p2, FORWARD, is_white) & empty;

    const Bitboard capL = BB_SHIFT_FORWARD_LEFT (pawns, is_white) & ~BOARD_FILE[FILE_H];
    const Bitboard capR = BB_SHIFT_FORWARD_RIGHT(pawns, is_white) & ~BOARD_FILE[FILE_A];

    const Bitboard promo = PROMOTION_ROW[is_white];
    const Bitboard cl = capL & enemy->side_all;
    const Bitboard cr = capR & enemy->side_all;

    results[PROMO_PUSH] = results[PUSH1] & promo;
    results[PUSH1]     &= ~promo;
    results[PROMO_CAPL] = cl & promo;
    results[CAPL]       = cl & ~promo;
    results[PROMO_CAPR] = cr & promo;
    results[CAPR]       = cr & ~promo;

    results[EPL] = 0ULL;
    results[EPR] = 0ULL;

    if (cb->ep_square != SQ_NONE) [[unlikely]] {
        const square king_pos = __builtin_ctzll(player->king);
        const square cap_sq   = cb->ep_square + color_dir(FORWARD, !is_white);
        bool pinned_ep = false;

        // EP-Pin (beide Bauern verschwinden von der Reihe) nur möglich, wenn der König auf derselben Reihe steht
        if ((king_pos >> 3) == (cap_sq >> 3)) {
            const Bitboard cap_bb = 1ULL << cap_sq;
            Bitboard line = attackers_straight_ray_pawns(cb, player, enemy, king_pos, player->king)
                            & ROWS[king_pos >> 3];
            while (line) {
                const square a = pop_lsb(line);
                const Bitboard between = SQUARES_IN_BETWEEN[a][king_pos];
                if ((cap_bb & between) &&
                    __builtin_popcountll(between & (player->pawns | enemy->pawns)) == 2) {
                    pinned_ep = true;
                    break;
                }
            }
        }
        if (!pinned_ep) {
            const Bitboard ep_bb = 1ULL << cb->ep_square;
            results[EPL] = capL & ep_bb;
            results[EPR] = capR & ep_bb;
        }
    }
}

static inline void find_pawn_moves(MoveStacks* moves, chess_board* cb, one_side* player, one_side* enemy)
{
    const Bitboard empty = ~cb->complete_board;
    Bitboard pinned_pawns = player->pawns & cb->pinned_pieces;
    const Bitboard free_pawns = player->pawns & ~pinned_pawns;

    Bitboard results[RESULT_COUNT];   // wird vollständig gesetzt, keine Nullinitialisierung nötig
    find_different_pawn_moves(free_pawns, empty, player, enemy, cb, results);

    if (!cb->attack_count) {
        if (pinned_pawns) [[unlikely]] {
            const square ksq = __builtin_ctzll(player->king);
            Bitboard tmp[RESULT_COUNT];
            while (pinned_pawns) {
                const square ps = pop_lsb(pinned_pawns);
                const Bitboard allowed = SQUARES_ON_THE_LINE[ps][ksq];
                find_different_pawn_moves(1ULL << ps, empty, player, enemy, cb, tmp);
                for (int i = 0; i < RESULT_COUNT; ++i) results[i] |= (allowed & tmp[i]);
            }
        }
    } else {
        Bitboard mask = cb->attack_defend_squares;
        // Schach durch Doppelschritt-Bauern: EP-Schlag löst das Schach auf
        if (cb->ep_square != SQ_NONE) {
            const Bitboard cap_bb = 1ULL << (cb->ep_square + color_dir(FORWARD, !cb->whites_turn));
            if (mask & cap_bb) mask |= 1ULL << cb->ep_square;
        }
        for (int i = 0; i < RESULT_COUNT; ++i) results[i] &= mask;
    }

    add_all_pawn_moves(results, moves, cb->whites_turn);
}

// ============================================================
//  Springer / Läufer / Türme / Damen
// ============================================================
static inline void find_knight_moves(MoveStacks* moves, chess_board* cb, one_side* player, one_side* enemy)
{
    Bitboard knights = player->knights & ~cb->pinned_pieces;   // gepinnte Springer können nie ziehen
    const Bitboard free_sq = ~player->side_all;
    const Bitboard mask = cb->attack_count ? cb->attack_defend_squares : ~0ULL;

    while (knights) {
        const square s = pop_lsb(knights);
        add_normal_moves(s, KNIGHT_LOOKUP_TABLE[s] & free_sq & mask, enemy->side_all, moves);
    }
}

static inline void find_slider_moves(MoveStacks* moves, chess_board* cb, one_side* player, one_side* enemy,
                                     Bitboard pieces, const MagicTableEntry* table)
{
    const bool in_check    = cb->attack_count != 0;
    const square king_sq   = __builtin_ctzll(player->king);
    const Bitboard free_sq = ~player->side_all;

    while (pieces) {
        const square s = pop_lsb(pieces);
        Bitboard dest = sliding_magic(s, cb->complete_board, table) & free_sq;

        if ((1ULL << s) & cb->pinned_pieces) {
            if (in_check) continue;
            dest &= SQUARES_ON_THE_LINE[s][king_sq];
        } else if (in_check) {
            dest &= cb->attack_defend_squares;
        }
        add_normal_moves(s, dest, enemy->side_all, moves);
    }
}



// ============================================================
//  König / Rochade über Angriffskarte
// ============================================================
static inline Bitboard enemy_attack_map(const chess_board* cb, const one_side* us, const one_side* them)
{
    const Bitboard occ = cb->complete_board & ~us->king;
    Bitboard att = KING_MOVES_MASK[__builtin_ctzll(them->king)]
                 | pawn_attacks_bb(them->pawns, !cb->whites_turn);

    Bitboard b = them->knights;
    while (b) att |= KNIGHT_LOOKUP_TABLE[pop_lsb(b)];

    b = them->bishop | them->queen;
    while (b) att |= bishop_attacks(pop_lsb(b), occ);

    b = them->rooks | them->queen;
    while (b) att |= rook_attacks(pop_lsb(b), occ);

    return att;
}

extern const CastlingRights CASTLING_FLAG[2][2] = {
    { BLACK_KING_SIDE,  BLACK_QUEEN_SIDE },
    { WHITE_KING_SIDE,  WHITE_QUEEN_SIDE }
};

static inline void add_castling(MoveStacks* moves, chess_board* board, square king_pos,
                                bool is_white, Bitboard enemy_attacks)
{
    if (board->attack_count || !(board->castling_rights & (is_white ? WHITE_CASTLING : BLACK_CASTLING)))
        return;

    Move* n = moves->normal_end;
    for (int cs = 0; cs < 2; ++cs) {
        if (!(board->castling_rights & CASTLING_FLAG[is_white][cs])) continue;
        if (board->complete_board & CASTLE_FREE[is_white][cs])       continue;
        if (CASTLE_SAVE[is_white][cs] & enemy_attacks)               continue;
        *n++ = Move(king_pos, CASTLE_TO[is_white][cs], CASTLING);
    }
    moves->normal_end = n;
}

// ============================================================
//  Generator
// ============================================================
static inline void gen_moves(MoveStacks* ms, chess_board* cb, one_side* us, one_side* them)
{
    const square k = __builtin_ctzll(us->king);

    find_pin_information(cb, us, them, k);

    const Bitboard att  = enemy_attack_map(cb, us, them);
    const Bitboard safe = KING_MOVES_MASK[k] & ~us->side_all & ~att;
    add_normal_moves(k, safe, them->side_all, ms);
    add_castling(ms, cb, k, cb->whites_turn, att);

    if (cb->attack_count >= 2) return;

    if (cb->attack_count == 1)
        cb->attack_defend_squares = SQUARES_IN_BETWEEN[k][__builtin_ctzll(cb->attacking_pieces)];

    find_knight_moves(ms, cb, us, them);
    find_pawn_moves  (ms, cb, us, them);
    find_slider_moves(ms, cb, us, them, us->bishop | us->queen, BISHOP_MAGIC);
    find_slider_moves(ms, cb, us, them, us->rooks  | us->queen, ROOK_MAGIC);
}

void find_all_moves(MoveStacks* ms, chess_board* cb)
{
    if (cb->whites_turn) gen_moves(ms, cb, &cb->white, &cb->black);
    else                 gen_moves(ms, cb, &cb->black, &cb->white);
}

// ============================================================
//  make / undo
// ============================================================
void make_move(chess_board* cb, Move m, StateInfo& st)
{
    const bool white = cb->whites_turn;
    one_side& us   = white ? cb->white : cb->black;
    one_side& them = white ? cb->black : cb->white;

    const square from = m.from_sq();
    const square to   = m.to_sq();
    const int flag    = m.move_flag();

    st.epSquare = cb->ep_square;
    st.castling = cb->castling_rights;
    st.rule50   = cb->halve_move_counter;
    st.captured = NO_PIECE_TYPE;

    const PieceType moving = piece_on(us, from);
    st.moving = moving;

    // Schlagen: piece_on nur, wenn dort überhaupt eine gegnerische Figur steht
    if (flag == 2) [[unlikely]] {
        st.captured = PAWN;
        remove_piece(them, PAWN, to + (white ? -8 : 8));
    } else if (them.side_all & (1ULL << to)) {
        const PieceType cap = piece_on(them, to);
        st.captured = cap;
        remove_piece(them, cap, to);
    }

    // Ziehen / Umwandeln
    if (flag == 1) [[unlikely]] {
        remove_piece(us, PAWN, from);
        add_piece(us, (PieceType)m.promo_piece(), to);
    } else {
        move_piece(us, moving, from, to);
    }

    // Rochade: Turm-Felder direkt aus to/from
    if (flag == 3) [[unlikely]] {
        if (to > from) move_piece(us, ROOK, to + 1, to - 1);   // kurz: h->f
        else           move_piece(us, ROOK, to - 2, to + 1);   // lang: a->d
    }

    cb->castling_rights = CastlingRights(cb->castling_rights & CASTLE_MASK[from] & CASTLE_MASK[to]);

    cb->ep_square = SQ_NONE;
    if (moving == PAWN && abs(to - from) == 16) {
        const square ep = (from + to) / 2;
        if (PAWN_ATTACK_LOOKUP_TABLE[white][ep] & them.pawns)
            cb->ep_square = ep;
    }

    cb->halve_move_counter = (moving == PAWN || st.captured != NO_PIECE_TYPE) ? 0 : cb->halve_move_counter + 1;
    if (!white) cb->full_move_counter++;

    cb->complete_board = cb->white.side_all | cb->black.side_all;
    cb->whites_turn    = !white;
}

void undo_move(chess_board* cb, Move m, const StateInfo& st)
{
    cb->whites_turn ^= 1;
    const bool white = cb->whites_turn;

    one_side& us   = white ? cb->white : cb->black;
    one_side& them = white ? cb->black : cb->white;

    const square from = m.from_sq();
    const square to   = m.to_sq();
    const int flag    = m.move_flag();

    cb->ep_square          = st.epSquare;
    cb->castling_rights    = st.castling;
    cb->halve_move_counter = st.rule50;
    if (!white) cb->full_move_counter--;

    if (flag == 3) [[unlikely]] {
        if (to > from) move_piece(us, ROOK, to - 1, to + 1);
        else           move_piece(us, ROOK, to + 1, to - 2);
    }

    if (flag == 1) [[unlikely]] {
        remove_piece(us, (PieceType)m.promo_piece(), to);
        add_piece(us, PAWN, from);
    } else {
        move_piece(us, st.moving, to, from);
    }

    if (st.captured != NO_PIECE_TYPE) {
        if (flag == 2) add_piece(them, PAWN, to + (white ? -8 : 8));
        else           add_piece(them, st.captured, to);
    }

    cb->complete_board = cb->white.side_all | cb->black.side_all;
}

// ============================================================
//  Perft
// ============================================================
uint64_t perft(chess_board* board, int depth)
{
    if (depth == 0) return 1;

    MoveStacks moves;
    find_all_moves(&moves, board);

    if (depth == 1)                                   // Bulk Counting
        return moves.normal_size() + moves.capture_size();

    uint64_t nodes = 0;
    StateInfo st;

    for (Move* m = moves.capture_moves; m != moves.capture_end; ++m) {
        chess_board next = *board;
        make_move(&next, *m, st);
        nodes += perft(&next, depth - 1);
    }
    for (Move* m = moves.normal_moves; m != moves.normal_end; ++m) {
        chess_board next = *board;
        make_move(&next, *m, st);
        nodes += perft(&next, depth - 1);
    }
    return nodes;
}

// try_all_moves unverändert

std::map<std::string, uint64_t> try_all_moves(
    chess_board* board,
    int depth)
{
    std::map<std::string, uint64_t> result;

    MoveStacks moves;
    find_all_moves(&moves, board);

    for(Move* m = moves.capture_moves; m != moves.capture_end; ++m){
        std::string move = m->move_to_string(board->whites_turn);

        StateInfo st;
        make_move(board, *m, st);

        result[move] = perft(board, depth - 1);

        undo_move(board, *m, st);
    }

    for(Move* m = moves.normal_moves; m != moves.normal_end; ++m){
        std::string move = m->move_to_string(board->whites_turn);

        StateInfo st;
        make_move(board, *m, st);

        result[move] = perft(board, depth - 1);

        undo_move(board, *m, st);
    }

    return result;
}



int msb_index(Bitboard bb) {
    return 63 - __builtin_clzll(bb);
}

void set_index_zero(Bitboard* bitboard, Bitboard index){
    *bitboard &= ~(1ULL << index);
}

void set_index_one(Bitboard* bitboard, Bitboard index) {
    *bitboard |= (1ULL << index);
}

bool is_in_check(chess_board* cb)
{
    one_side& us   = cb->whites_turn ? cb->white : cb->black;
    one_side& them = cb->whites_turn ? cb->black : cb->white;
    if (us.king == 0) return false;
    return !is_save_square(cb, &us, &them, __builtin_ctzll(us.king), us.king);
}



bool is_save_square(chess_board* chess_board, one_side* player, one_side* enemy, square pos_ind, Bitboard original_square){
    // Checks if a pos is attacked by a piece
    
    if(get_straight_attackers(chess_board, player, enemy, pos_ind, original_square)){
        return false;
    }

    if(get_diagonal_attackers(chess_board, player, enemy, pos_ind, original_square)){
        return false;
    }

    if(KNIGHT_LOOKUP_TABLE[pos_ind] & enemy->knights){
        return false;
    }

    if(PAWN_ATTACK_LOOKUP_TABLE[chess_board->whites_turn][pos_ind]&enemy->pawns){
        return false;
    }

    if(KING_MOVES_MASK[pos_ind] & enemy->king){
        return false;
    }

    return true;
}



void setup_fen_position(chess_board& board, const std::string& fen)
{
    // Clear board
    board.white = {};
    board.black = {};
    board.complete_board = 0ULL;
    board.castling_rights = NO_CASTLING;
    board.ep_square = -1;
    board.halve_move_counter = 0;
    board.full_move_counter = 1;

    std::istringstream ss(fen);

    std::string placement, active, castling, enpassant;
    int halfmove = 0, fullmove = 1;

    ss >> placement >> active >> castling >> enpassant >> halfmove >> fullmove;

    // --- Piece placement ---
    int square = 56;   // a8

    for (char c : placement)
    {
        if (c == '/')
        {
            square -= 16;   // next rank down
            continue;
        }

        if (std::isdigit(c))
        {
            square += (c - '0');
            continue;
        }

        Bitboard bit = 1ULL << square;

        switch (c)
        {
            case 'P': board.white.pawns   |= bit; break;
            case 'N': board.white.knights |= bit; break;
            case 'B': board.white.bishop  |= bit; break;
            case 'R': board.white.rooks   |= bit; break;
            case 'Q': board.white.queen   |= bit; break;
            case 'K': board.white.king    |= bit; break;

            case 'p': board.black.pawns   |= bit; break;
            case 'n': board.black.knights |= bit; break;
            case 'b': board.black.bishop  |= bit; break;
            case 'r': board.black.rooks   |= bit; break;
            case 'q': board.black.queen   |= bit; break;
            case 'k': board.black.king    |= bit; break;
        }

        square++;
    }

    // --- Active color ---
    board.whites_turn = (active == "w");

    // --- Castling rights ---
    board.castling_rights = NO_CASTLING;

    if (castling != "-")
    {
        for (char c : castling)
        {
            switch (c)
            {
                case 'K':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | WHITE_KING_SIDE);
                    break;

                case 'Q':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | WHITE_QUEEN_SIDE);
                    break;

                case 'k':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | BLACK_KING_SIDE);
                    break;

                case 'q':
                    board.castling_rights =
                        CastlingRights(board.castling_rights | BLACK_QUEEN_SIDE);
                    break;
            }
        }
    }

    // --- En-passant square ---
    if (enpassant != "-" && enpassant.size() == 2)
    {
        int file = enpassant[0] - 'a';
        int rank = enpassant[1] - '1';
        board.ep_square = rank * 8 + file;
    }
    else
    {
        board.ep_square = -1;
    }

    // --- Clocks ---
    board.halve_move_counter  = halfmove;
    board.full_move_counter = fullmove;

    // --- Derived bitboards ---
    board.white.update_side();
    board.black.update_side();
    board.complete_board = board.white.side_all | board.black.side_all;
}

std::string board_to_fen(const chess_board& board)
{
    std::string fen;

    // --- Piece placement ---
    for (int rank = 7; rank >= 0; --rank)
    {
        int empty = 0;

        for (int file = 0; file < 8; ++file)
        {
            int square = rank * 8 + file;
            Bitboard bit = 1ULL << square;
            char piece = 0;

            // White pieces
            if (board.white.pawns   & bit) piece = 'P';
            else if (board.white.knights & bit) piece = 'N';
            else if (board.white.bishop  & bit) piece = 'B';
            else if (board.white.rooks   & bit) piece = 'R';
            else if (board.white.queen   & bit) piece = 'Q';
            else if (board.white.king    & bit) piece = 'K';

            // Black pieces
            else if (board.black.pawns   & bit) piece = 'p';
            else if (board.black.knights & bit) piece = 'n';
            else if (board.black.bishop  & bit) piece = 'b';
            else if (board.black.rooks   & bit) piece = 'r';
            else if (board.black.queen   & bit) piece = 'q';
            else if (board.black.king    & bit) piece = 'k';

            if (piece)
            {
                if (empty > 0)
                {
                    fen += std::to_string(empty);
                    empty = 0;
                }
                fen += piece;
            }
            else
            {
                ++empty;
            }
        }

        if (empty > 0)
            fen += std::to_string(empty);

        if (rank > 0)
            fen += '/';
    }

    // --- Active color ---
    fen += ' ';
    fen += (board.whites_turn ? 'w' : 'b');

    // --- Castling rights ---
    fen += ' ';
    std::string castling;

    if (board.castling_rights & WHITE_KING_SIDE)  castling += 'K';
    if (board.castling_rights & WHITE_QUEEN_SIDE) castling += 'Q';
    if (board.castling_rights & BLACK_KING_SIDE)  castling += 'k';
    if (board.castling_rights & BLACK_QUEEN_SIDE) castling += 'q';

    fen += (castling.empty() ? "-" : castling);

    // --- En-passant square ---
    fen += ' ';
    if (board.ep_square >= 0)
    {
        int file = board.ep_square % 8;
        int rank = board.ep_square / 8;
        fen += static_cast<char>('a' + file);
        fen += static_cast<char>('1' + rank);
    }
    else
    {
        fen += "-";
    }

    // --- Clocks ---
    fen += ' ';
    fen += std::to_string(board.halve_move_counter);
    fen += ' ';
    fen += std::to_string(board.full_move_counter);

    return fen;
}

