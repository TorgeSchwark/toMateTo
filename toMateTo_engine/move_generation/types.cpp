#include "types.h"

const int8_t FORWARD[2] = {
    -8, // BLACK
     8  // WHITE
};

const int8_t FORWARD_LEFT[2] = {
    -7, // BLACK (from black POV: capture left = south-east)
     7  // WHITE (north-west)
};

const int8_t FORWARD_RIGHT[2] = {
    -9, // BLACK (south-west)
     9  // WHITE (north-east)
};

const int8_t DOUBLE_FORWARD[2] = {
    -16, // BLACK
     16  // WHITE
};