#ifndef MOVE_STACK
#define MOVE_STACK

#include <cstdint>
#include <string>    
#include <iostream>   

#include "toMateTo_engine/move_generation/types.h"

// TODO change to that structure:
// A move needs 16 bits to be stored
//
// bit  0- 5: destination square (from 0 to 63)
// bit  6-11: origin square (from 0 to 63)
// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
// NOTE: en passant bit is set only when a pawn can be captured


inline std::string square_to_string(int index) {
    char file = 'a' + (index % 8);
    char rank = '1' + (index / 8);
    return std::string() + file + rank;
}


struct Move{

    Move() = default;
    constexpr Move(std::uint16_t d) :
        move(d) {};

    constexpr Move(square from, square to)
        : move((from << 6) | to) {}

    constexpr square from_sq(const Move& m) {
        return square((m.move >> 6) & 0x3F);
    }

    constexpr square to_sq(const Move& m) {
        return square(m.move & 0x3F);
    }
    
    inline std::string move_to_string() {
        square from = from_sq(move);
        square to   = to_sq(move);

        return square_to_string(from) + " -> " + square_to_string(to);
    }


    int16_t move;
};

inline constexpr Bitboard KNIGHT_MOVES[4] = {17,15,10,6};
inline constexpr int8_t NORMAL_MOVE = 1;








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