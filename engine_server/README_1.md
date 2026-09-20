# ToMateTo im Browser

## Dateien

```
engine_server.h / engine_server.cpp   -> in dein Projekt kopieren (neben engine_match.*)
web/index.html                        -> die Webseite
```

## In main() einbauen

```cpp
#include "engine_server.h"

int main(int argc, char** argv)
{
    // ... deine normale Initialisierung (Zobrist-Tabellen, Attack-Tables, TT ...)

    if (argc > 1 && std::string(argv[1]) == "--server")
    {
        int port = argc > 2 ? std::atoi(argv[2]) : 8080;
        run_web_server(port, "web", 4);   // 4 = alle vier Kerne
        return 0;
    }

    // ... dein bisheriges Verhalten
}
```

## Bauen (auf dem Arduino selbst)

```bash
g++ -O3 -std=c++17 -pthread *.cpp -o ToMateTo
```

(`-march=native` nur nutzen, wenn du direkt auf dem Board kompilierst.)

## Starten

```bash
./ToMateTo --server 8080
```

Dann im Browser (Handy/PC im selben WLAN): `http://<IP-des-Arduino>:8080`

## API (falls du selbst testen willst)

| Route | Zweck |
|---|---|
| `/api/info?fen=...` | legale Züge + Status einer Stellung |
| `/api/move?fen=...&move=e2e4` | Zug ausführen, neue Stellung zurück |
| `/api/engine?fen=...&time=2` | ToMateTo zieht (Zeit in Sekunden) |
