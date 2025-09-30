#ifndef MOVE_STACK
#define MOVE_STACK

#include <cstdint>


struct move_stack{
    uint8_t moves[500];
    uint8_t move_counter;
    uint8_t move_types[500];
};


#endif