#ifndef MOVE_STACK
#define MOVE_STACK

#include <cstdint>
#include <string>    
#include <iostream>   

#include "toMateTo_engine/move_generation/types.h"

inline constexpr Bitboard KNIGHT_MOVES[4] = {17,15,10,6};
inline constexpr int8_t NORMAL_MOVE = 1;


inline std::string square_to_string(int index) {
    char file = 'a' + (index % 8);
    char rank = '1' + (index / 8);
    return std::string() + file + rank;
}

inline std::string piece_to_string(PieceType piece) {
    switch (piece) {
        case PAWN: return "Pawn";
        case BISHOP: return "Bishop";
        case KNIGHT: return "Knight";
        case ROOK: return "Rook";
        case QUEEN: return "Queen";
        case KING: return "King";
        default: return "Unknown";
    }
}

inline std::string move_type_to_string(int type) {
    switch (type) {
        case 1: return "Normal";
        case 2: return "Capture";
        case 3: return "Promotion";
        default: return "Other";
    }
}



struct move_stack{
    Bitboard moves[500];
    int move_counter = 0;
    Bitboard move_types[500];

    inline void add_move(Bitboard from, Bitboard to, Bitboard piece, Bitboard move_type){

        moves[move_counter] = from;
        moves[move_counter+1] = to;
        moves[move_counter+2] = piece;
        moves[move_counter+3] = move_type;
        move_counter += 4;
    };

    void clear(){
        move_counter = 0;
    }
    // Diese Methode in deine `move_stack`-Struktur integrieren:
    void print_moves() const {
        std::cout << "=== Move Stack ===" << std::endl;
        for (int i = 0; i < move_counter; i += 4) {
            int from = __builtin_ctzll(moves[i]);
            int to = __builtin_ctzll(moves[i + 1]);
            PieceType piece = static_cast<PieceType>(moves[i + 2]);
            int type = moves[i + 3];

            std::cout << "Move " << (i / 4) << ": "
                    << square_to_string(from) << " -> " << square_to_string(to)
                    << "  (Piece: " << piece_to_string(piece)
                    << ", Type: " << move_type_to_string(type) << ")"
                    << std::endl;
        }
    }
};





#endif