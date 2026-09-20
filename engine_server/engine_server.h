#pragma once
#include <string>

#include "testing/engine_match.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Startet den ToMateTo-Webserver (blockiert dauerhaft).
//   port      : z.B. 8080
//   web_dir   : Ordner, in dem index.html liegt (z.B. "web")
//   max_engines: wie viele Engine-Berechnungen gleichzeitig laufen dürfen
//                (= genutzte CPU-Kerne, beim Arduino UNO Q also 4)
void run_web_server(int port,
                    const std::string& web_dir,
                    int max_engines = 4);
