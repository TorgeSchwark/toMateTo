#include "toMateTo_engine/table_generation/knight_tables.h"

Bitboard KNIGHT_LOOKUP_TABLE[64]; 

void generate_knight_tables(Bitboard lookup_table[64]) {
    const int directions[8][2] = {
        {-2, -1}, {-1, -2}, {-2,  1}, {1, -2},
        { 2, -1}, {-1,  2}, {1,  2},  {2,  1}
    };

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            uint64_t bitboard = 0ULL;
            for (int d = 0; d < 8; ++d) {
                int nx = x + directions[d][0];
                int ny = y + directions[d][1];
                if (nx >= 0 && nx < 8 && ny >= 0 && ny < 8) {
                    int target_index = nx * 8 + ny;
                    bitboard |= (1ULL << target_index);
                }
            }
            lookup_table[x * 8 + y] = bitboard;
        }
    }
}

void init_knight_table() {
    generate_knight_tables(KNIGHT_LOOKUP_TABLE);
}
