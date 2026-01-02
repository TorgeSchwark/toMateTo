#ifndef MOVE_STACK
#define MOVE_STACK

#include <cstdint>
#include <string>    
#include <iostream>   



const int64_t KNIGHT_MOVES[4] = {17,15,10,6};  
const int64_t KNIGHT = 3;
const int64_t BISHOP = 2;
const int64_t PAWN = 1;
const int64_t ROOK = 4;
const int64_t QUEEN = 5;
const int64_t KING = 6;
const int64_t NORMAL_MOVE = 1;



inline std::string square_to_string(int index) {
    char file = 'a' + (index % 8);
    char rank = '1' + (index / 8);
    return std::string() + file + rank;
}

inline std::string piece_to_string(int piece) {
    switch (piece) {
        case 1: return "Pawn";
        case 2: return "Bishop";
        case 3: return "Knight";
        case 4: return "Rook";
        case 5: return "Queen";
        case 6: return "King";
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
    int64_t moves[500];
    int move_counter = 0;
    int64_t move_types[500];

    void add_move(int64_t from, int64_t to, int64_t piece, int64_t move_type){

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
            int piece = moves[i + 2];
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