#ifndef ENGINE_MATCH
#define ENGINE_MATCH

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <unistd.h>
#include <signal.h>
#include <thread>
#include <vector>
#include <mutex>
#include <unistd.h>

#include "toMateTo/toMateTo.h"

void run_engine_match_ST(
    double engine_time,
    int stockfish_elo);
    
void run_engine_match(
    double engine_time,
    int stockfish_elo
);



#endif