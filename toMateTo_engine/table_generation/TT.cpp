#include "TT.h"
#include "toMateTo_engine/toMateTo/profiler.h"



#include <random>

// ---------------------------------------------------------
// Zobrist tables
// ---------------------------------------------------------

uint64_t zobrist_piece[2][6][64];
uint64_t zobrist_castling[16];
uint64_t zobrist_ep[64];
uint64_t zobrist_side;

// ---------------------------------------------------------
// Transposition Table
// ---------------------------------------------------------

thread_local std::vector<TTEntry> transposition_table(TT_SIZE);

// ---------------------------------------------------------
// Random number generator
// ---------------------------------------------------------

static uint64_t random_uint64()
{
    static std::mt19937_64 rng(123456789);
    return rng();
}

// ---------------------------------------------------------
// Initialize Zobrist tables
// ---------------------------------------------------------

int zobrist_piece_index(PieceType piece)
{
    switch (piece)
    {
        case PAWN:   return 0;
        case KNIGHT: return 1;
        case BISHOP: return 2;
        case ROOK:   return 3;
        case QUEEN:  return 4;
        case KING:   return 5;
        default:     return -1;
    }
}

void init_zobrist()
{
    for (int color = 0; color < 2; color++)
    {
        for (int piece = 0; piece < 6; piece++)
        {
            for (int square = 0; square < 64; square++)
            {
                zobrist_piece[color][piece][square] = random_uint64();
            }
        }
    }

    for (int i = 0; i < 16; i++)
    {
        zobrist_castling[i] = random_uint64();
    }

    for (int i = 0; i < 64; i++)
    {
        zobrist_ep[i] = random_uint64();
    }

    zobrist_side = random_uint64();
}

static uint64_t hash_side(
    const one_side& side,
    int color)
{
    uint64_t hash = 0;

    Bitboard pieces[6] =
    {
        side.pawns,
        side.knights,
        side.bishop,
        side.rooks,
        side.queen,
        side.king
    };

    for (int piece = BISHOP; piece <= PAWN; piece++)
    {
        if(piece == NO_PIECE_TYPE){
            continue;
        }
        int index = zobrist_piece_index((PieceType) piece);

        Bitboard bb = pieces[index];

        while (bb)
        {
            int square = __builtin_ctzll(bb);


            hash ^= zobrist_piece[color][index][square];

            bb &= bb - 1;
        }
    }

    return hash;
}

uint64_t calculate_hash(chess_board* board)
{

    Profiler::Scope profile("calculate_hash");
    uint64_t hash = 0;

    // White pieces
    hash ^= hash_side(board->white, WHITE);

    // Black pieces
    hash ^= hash_side(board->black, BLACK);

    // Side to move
    if (!board->whites_turn)
    {
        hash ^= zobrist_side;
    }

    // Castling rights
    hash ^= zobrist_castling[board->castling_rights];

    // En passant
    if (board->ep_square != SQ_NONE)
    {
        hash ^= zobrist_ep[board->ep_square];
    }

    return hash;
}