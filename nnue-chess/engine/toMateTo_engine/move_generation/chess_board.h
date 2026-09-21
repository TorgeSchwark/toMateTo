#ifndef CHESS_BOARD
#define CHESS_BOARD

#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <bit>
#include <map>
#include <array>
#include <cstdlib>   // abs

#include "toMateTo_engine/move_generation/move_stack.h"
#include "toMateTo_engine/table_generation/knight_tables.h"
#include "toMateTo_engine/table_generation/magic_gen.h"
#include "toMateTo_engine/table_generation/magic_king_tables.h"
#include "toMateTo_engine/move_generation/types.h"


void set_index_zero(Bitboard* bitboard, Bitboard index); 
int msb_index(Bitboard bb);
void set_index_one(Bitboard* bitboard, Bitboard index);
// chess_board.h: ersetzt "extern int RESULT_COUNT;"
inline constexpr int RESULT_COUNT = 9;

static inline int8_t color_dir(int8_t magnitude, bool white) {
    return white ? magnitude : -magnitude;
}

#define BB_SHIFT(bb, s, color) \
    ((color) ? ((bb) << (s)) : ((bb) >> (s)))

#define BB_SHIFT_FORWARD_RIGHT(bb, color) \
    ((color) ? ((bb) << 9) : ((bb) >> 7))

#define BB_SHIFT_FORWARD_LEFT(bb, color) \
    ((color) ? ((bb) << 7) : ((bb) >> 9))


inline constexpr std::array<uint8_t, 64> CASTLE_MASK = [] {
    std::array<uint8_t, 64> a{};
    for (auto& x : a) x = 15;
    a[A1] = 13; a[H1] = 14; a[E1] = 12;   // K=1, Q=2
    a[A8] = 7;  a[H8] = 11; a[E8] = 3;    // k=4, q=8
    return a;
}();
    
struct one_side
{
    union {
        struct { Bitboard bishop, knights, rooks, queen, king, unused_, pawns; };
        Bitboard by_type[7];
    };
    Bitboard side_all;
    Bitboard save_king_squares;   // kann später ganz raus

    inline void update_side() { side_all = knights | pawns | rooks | bishop | king | queen; }

    void setup_side(bool is_white) {
    if (is_white) {
        pawns = 0x000000000000FF00 & ~((1ULL << 9)  |  // b2
                               (1ULL << 11) |  // d2
                               (1ULL << 12) |  // e2
                               (1ULL << 14));  // g2
        rooks  = 0x0000000000000081;  // a1, h1 (Bits 0, 7)
        knights= 0x0000000000000042;  // b1, g1 (Bits 1, 6)
        bishop = 0x0000000000000024;  // c1, f1 (Bits 2, 5)
        queen  = 0x0000000000000008;  // d1 (Bit 3)
        king   = 0x0000000000000010;  // e1 (Bit 4)
    } else {
        pawns  = 0x00FF000000000000;  // Bits 48–55
        rooks  = 0x8100000000000000;  // a8, h8 (Bits 56, 63)
        knights= 0x4200000000000000;  // b8, g8 (Bits 57, 62)
        bishop = 0x2400000000000000;  // c8, f8 (Bits 58, 61)
        queen  = 0x0800000000000000;  // d8 (Bit 59)
        king   = 0x1000000000000000;  // e8 (Bit 60)
    }

    update_side(); // setzt side_all = kombiniertes Bitboard
}


};


inline void remove_piece(one_side& s, PieceType pt, square sq) {
    const Bitboard b = 1ULL << sq;
    s.by_type[pt] &= ~b;
    s.side_all    &= ~b;
}
inline void add_piece(one_side& s, PieceType pt, square sq) {
    const Bitboard b = 1ULL << sq;
    s.by_type[pt] |= b;
    s.side_all    |= b;
}
inline void move_piece(one_side& s, PieceType pt, square from, square to) {
    const Bitboard ft = (1ULL << from) | (1ULL << to);
    s.by_type[pt] ^= ft;
    s.side_all    ^= ft;
}


struct chess_board
{
    one_side white;
    one_side black;

    bool whites_turn;
    square ep_square;

    int16_t halve_move_counter;
    int16_t full_move_counter;

    void setup_chess_board(){
        white.setup_side(true);
        black.setup_side(false);

        update_board();
        whites_turn = true;
    };

    void print_board() {
        const char empty = '.';

        // Array für jedes Feld auf dem Board
        char board[64];
        for (int i = 0; i < 64; ++i) board[i] = empty;

        // writes the symbol of the Piece on the correct position of the String representation
        auto set_piece = [&](Bitboard bitboard, char piece_symbol) {
            Bitboard copy_bitboard = bitboard;
            while(copy_bitboard){
                int index = msb_index(copy_bitboard);
                board[index] = piece_symbol;
                set_index_zero(&copy_bitboard, index);
            }
        };

        // Weiße Figuren
        set_piece(white.pawns,  'P');
        set_piece(white.knights,'N');
        set_piece(white.bishop, 'B');
        set_piece(white.rooks,  'R');
        set_piece(white.queen,  'Q');
        set_piece(white.king,   'K');

        // Schwarze Figuren
        set_piece(black.pawns,  'p');
        set_piece(black.knights,'n');
        set_piece(black.bishop, 'b');
        set_piece(black.rooks,  'r');
        set_piece(black.queen,  'q');
        set_piece(black.king,   'k');

        // Ausgabe des Boards: von oben nach unten (Rank 8 bis 1)
        for (int rank = 7; rank >= 0; --rank) {
            std::cout << rank + 1 << " ";
            for (int file = 0; file < 8; ++file) {
                std::cout << board[rank * 8 + file] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "  a b c d e f g h" << std::endl;
    }

    inline void update_white()
    {
        white.update_side();
    };

    inline void update_black()
    {
        black.update_side();
    };

    Bitboard complete_board;
    Bitboard pinned_pieces;
    // Pieces currently attacking the King
    Bitboard attacking_pieces;
    // Can be used if king is attacked only once to filter where pieces can go!
    Bitboard attack_defend_squares;

    int8_t attack_count;

    CastlingRights castling_rights;  


    inline void update_board()
    {
        update_black();
        update_white();
        complete_board =  white.side_all | black.side_all;
    }
};

std::map<std::string, uint64_t> try_all_moves(
    chess_board* board,
    int depth);

uint64_t perft(chess_board* board, int depth);

void make_move(chess_board* cb, Move m, StateInfo& st);

void undo_move(chess_board* cb, Move m, const StateInfo& st);

void find_all_moves(MoveStacks* move_stacks, chess_board* chess_board);

bool is_save_square(chess_board* chess_board, one_side* player, one_side* enemy, square pos_ind, Bitboard original_square);

inline square pop_lsb(Bitboard &board){
    square bishop_index = __builtin_ctzll(board);        // Get index of least significant bit
    board &= board - 1;
    return bishop_index;
};

inline int lsb(Bitboard b) {
    return __builtin_ctzll(b);
}

inline int msb(Bitboard b) {
    return 63 - __builtin_clzll(b);
}

inline PieceType piece_on(const one_side& s, square sq) {
    const Bitboard b = 1ULL << sq;
    if (s.pawns   & b) return PAWN;
    if (s.knights & b) return KNIGHT;
    if (s.bishop  & b) return BISHOP;
    if (s.rooks   & b) return ROOK;
    if (s.queen   & b) return QUEEN;
    if (s.king    & b) return KING;
    return NO_PIECE_TYPE;
}

inline Bitboard magic_lookup(Bitboard occ, const MagicTableEntry& m) {
    int index = (int)((occ * m.magic_number) >> (64 - m.relevant_bits));
    return m.attack_list[index];
}

inline Bitboard pins_magic(int square, Bitboard blocked, const MagicTableEntry table[]){
    return magic_lookup(blocked, table[square]);
}




inline Bitboard attackers_magic(int square, Bitboard attackers,const MagicTableEntry table[],const MagicTableEntry pattern[]){
    Bitboard masked = attackers & table[square].mask;
    return magic_lookup(masked, pattern[square]);
}

inline Bitboard get_straight_pins(one_side* enemy, one_side* player,int pos_ind, Bitboard attack_mask){
    Bitboard blocked = attack_mask & (enemy->knights | enemy->bishop | enemy->pawns| enemy->king | player->side_all);
    return pins_magic(pos_ind, blocked, PINNED_PIECES_ROOK_MAGIC);
}

inline Bitboard get_diagonal_pins(one_side* enemy, one_side* player, int pos_ind, Bitboard attack_mask){
    Bitboard blocked = attack_mask & (enemy->knights | enemy->rooks | enemy->pawns | enemy->king | player->side_all);
    return pins_magic(pos_ind, blocked, PINNED_PIECES_BISHOP_MAGIC);
}


inline Bitboard sliding_magic(int square, Bitboard occ, const MagicTableEntry table[], Bitboard blockers_mask = ~0ULL){
    return magic_lookup(occ & table[square].mask, table[square]) & blockers_mask;
}

bool is_in_check(chess_board* chess_board);

inline Bitboard get_squares_til_straight_attacker(one_side* enemy,
                                           square pos_ind ) {

    Bitboard attackers_mask = 0LL;
    Bitboard straight_sliders = enemy->queen | enemy->rooks;

    for (int dir = NORTH; dir <= WEST; ++dir) {

        Bitboard b = straight_sliders & DIRECTION_RAYS[pos_ind][dir];
        if (!b)
            continue;

        // nächster gegnerischer Slider in dieser Richtung
        int sq = (DIR_DELTA[dir] > 0) ? lsb(b) : msb(b);
        attackers_mask |= SQUARES_IN_BETWEEN[pos_ind][sq];
    }

    return attackers_mask;
}

inline Bitboard get_squares_til_diagonal_attacker(one_side* enemy,
                                           square pos_ind) {

    Bitboard attackers_mask = 0LL;
    Bitboard diagonal_sliders = enemy->queen | enemy->bishop;

    for (int dir = NE; dir <= SW; ++dir) {

        Bitboard b = diagonal_sliders & DIRECTION_RAYS[pos_ind][dir];
        if (!b)
            continue;

        // nächster gegnerischer Slider in dieser Richtung
        int sq = (DIR_DELTA[dir] > 0) ? lsb(b) : msb(b);

        attackers_mask |= SQUARES_IN_BETWEEN[pos_ind][sq];
    }

    return attackers_mask;
}

inline Bitboard bishop_magic(int square, const chess_board* board, const one_side* player){
    return sliding_magic(square, board->complete_board, BISHOP_MAGIC, ~player->side_all);
}

inline Bitboard bishop_magic_captures(int square, const chess_board* board, const one_side* enemy){
    return sliding_magic(square, board->complete_board, BISHOP_MAGIC, enemy->side_all);
}

inline Bitboard rook_magic(int square,const chess_board* board, const one_side* player){
    return sliding_magic(square, board->complete_board, ROOK_MAGIC, ~player->side_all);
}

inline Bitboard rook_magic_captures(int square,const chess_board* board, const one_side* enemy){
    return sliding_magic(square, board->complete_board, ROOK_MAGIC, enemy->side_all);
}

inline Bitboard bishop_magic_remove_original(int square, const chess_board* board, const one_side* player, Bitboard remove_mask){
    return sliding_magic(square, board->complete_board & (~remove_mask), BISHOP_MAGIC, ~player->side_all);
}

inline Bitboard bishop_attacks(int sq, Bitboard occ){
    return sliding_magic(sq, occ, BISHOP_MAGIC);
}

inline Bitboard rook_attacks(int sq, Bitboard occ){
    return sliding_magic(sq, occ, ROOK_MAGIC);
}

inline Bitboard pawn_attacks_bb(Bitboard pawns, bool white){
    const Bitboard notA = ~0x0101010101010101ULL;
    const Bitboard notH = ~0x8080808080808080ULL;
    return white ? (((pawns << 7) & notH) | ((pawns << 9) & notA))
                 : (((pawns >> 9) & notH) | ((pawns >> 7) & notA));
}

inline Bitboard rook_magic_remove_original(int square,const chess_board* board, const one_side* player, Bitboard remove_mask){
    return sliding_magic(square, board->complete_board & (~remove_mask), ROOK_MAGIC, ~player->side_all);
}

// finds the first attacker in a straight but ignoring Pawns!
inline Bitboard rook_magic_remove_original_ray_pawns(int square,const chess_board* board, const one_side* player, Bitboard remove_mask){
    return sliding_magic(square, board->complete_board & (~(remove_mask | board->white.pawns | board->black.pawns)), ROOK_MAGIC);
}

inline Bitboard get_diagonal_attackers(chess_board* chess_board, one_side* player, one_side* enemy, square pos_ind, Bitboard remove_mask){
    Bitboard diagonal_moves = bishop_magic_remove_original(pos_ind, chess_board, player, remove_mask);

    return (diagonal_moves & (enemy->bishop | enemy->queen));
}
inline Bitboard attackers_straight_ray_pawns(chess_board* chess_board, one_side* player, one_side* enemy, square pos_ind, Bitboard remove_mask){
    // für die attacks muss hier wirklich gefragt werden ob die figur eine gegnereischer Rook oder Queen
    Bitboard straight_moves = rook_magic_remove_original_ray_pawns(pos_ind, chess_board, player, remove_mask);
    // nur die auf der Geraden Linie 
    return (straight_moves & (enemy->rooks | enemy->queen));
}

inline Bitboard get_straight_attackers(chess_board* chess_board, one_side* player, one_side* enemy, square pos_ind, Bitboard remove_mask){
    // für die attacks muss hier wirklich gefragt werden ob die figur eine gegnereischer Rook oder Queen
    Bitboard straight_moves = rook_magic_remove_original(pos_ind, chess_board, player, remove_mask);

    return (straight_moves & (enemy->rooks | enemy->queen));
}

void setup_fen_position(chess_board& board, const std::string& fen);

std::string board_to_fen(const chess_board& board);

// ============================================================
//  Gemeinsame Movegen-Helfer (von mehreren .cpp-Dateien genutzt)
// ============================================================
inline void add_normal_moves(square from, Bitboard dest, Bitboard enemys, MoveStacks* moves){
    Bitboard caps  = dest & enemys;
    Bitboard quiet = dest & ~enemys;
    Move* n = moves->normal_end;
    Move* c = moves->capture_end;
    while (quiet) *n++ = Move(from, pop_lsb(quiet));
    while (caps)  *c++ = Move(from, pop_lsb(caps));
    moves->normal_end  = n;
    moves->capture_end = c;
}

inline Move* add_pawn_moves(Bitboard dest, Move* moves, int8_t offset, bool color){
    while (dest) {
        square to = pop_lsb(dest);
        *moves++ = Move(to - color_dir(offset, color), to);
    }
    return moves;
}

inline Move* add_prom(Bitboard dest, Move* moves, int8_t offset, bool color){
    while (dest) {
        square to   = pop_lsb(dest);
        square from = to - color_dir(offset, color);
        for (int8_t type = BISHOP; type <= QUEEN; type++)
            *moves++ = Move(from, to, PROMOTION, type);
    }
    return moves;
}

inline Move* add_ep(Bitboard dest, Move* moves, int8_t offset, bool color){
    if (dest) {
        square to = pop_lsb(dest);
        *moves++ = Move(to - color_dir(offset, color), to, EN_PASSANT);
    }
    return moves;
}

inline void add_all_pawn_moves(const Bitboard* r, MoveStacks* moves, bool color){
    Move* n = moves->normal_end;
    Move* c = moves->capture_end;
    n = add_pawn_moves(r[PUSH1],      n, FORWARD,              color);
    n = add_pawn_moves(r[PUSH2],      n, DOUBLE_FORWARD,       color);
    n = add_prom      (r[PROMO_PUSH], n, FORWARD,              color);
    c = add_pawn_moves(r[CAPL],       c, FORWARD_LEFT[color],  color);
    c = add_pawn_moves(r[CAPR],       c, FORWARD_RIGHT[color], color);
    c = add_ep        (r[EPL],        c, FORWARD_LEFT[color],  color);
    c = add_ep        (r[EPR],        c, FORWARD_RIGHT[color], color);
    c = add_prom      (r[PROMO_CAPL], c, FORWARD_LEFT[color],  color);
    c = add_prom      (r[PROMO_CAPR], c, FORWARD_RIGHT[color], color);
    moves->normal_end  = n;
    moves->capture_end = c;
}

constexpr std::array<Bitboard, 64> make_empty_rays(bool rook) {
    std::array<Bitboard, 64> t{};
    for (int s = 0; s < 64; ++s) {
        const int r = s / 8, f = s % 8;
        Bitboard b = 0;
        for (int d = 0; d < 4; ++d) {
            const int dr = rook ? (d == 0 ? 1 : d == 1 ? -1 : 0) : (d < 2 ? 1 : -1);
            const int df = rook ? (d == 2 ? 1 : d == 3 ? -1 : 0) : (d % 2 ? 1 : -1);
            for (int rr = r + dr, ff = f + df; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr += dr, ff += df)
                b |= 1ULL << (rr * 8 + ff);
        }
        t[s] = b;
    }
    return t;
}
inline constexpr std::array<Bitboard, 64> ROOK_EMPTY   = make_empty_rays(true);
inline constexpr std::array<Bitboard, 64> BISHOP_EMPTY = make_empty_rays(false);

inline void find_pin_information(chess_board* cb, one_side* us, one_side* them, square k)
{
    const Bitboard occ = cb->complete_board;
    const Bitboard own = us->side_all;
    const Bitboard nk  = ~us->king;

    Bitboard snipers =
        (ROOK_EMPTY[k]   & (them->rooks  | them->queen)) |
        (BISHOP_EMPTY[k] & (them->bishop | them->queen));

    Bitboard checkers = (KNIGHT_LOOKUP_TABLE[k] & them->knights)
                      | (PAWN_ATTACK_LOOKUP_TABLE[cb->whites_turn][k] & them->pawns);
    Bitboard pinned = 0ULL;

    while (snipers) {
        const square s = pop_lsb(snipers);
        const Bitboard between = SQUARES_IN_BETWEEN[k][s] & occ & ~(1ULL << s) & nk;
        if (!between)
            checkers |= 1ULL << s;
        else if (!(between & (between - 1)))
            pinned |= between & own;
    }

    cb->pinned_pieces    = pinned;
    cb->attacking_pieces = checkers;
    cb->attack_count     = (int8_t)__builtin_popcountll(checkers);
}

#endif 