#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>

#include "./move_generation/chess_board.h"
#include "./generation/knight_tables.h"
#include "./generation/magic_gen.h"
#include "./generation/king_tables.h"



// Hier deine includes für chess_board, move_stack, deine Funktionen etc.
// #include "chess.h"  (angenommen)

const int TILE_SIZE = 80;

int main() {
    std::cout << "Willkommen beim Schachspiel!" << std::endl;

    // ------------- Schachlogik initialisieren -------------
    chess_board chess_board;
    chess_board.setup_chess_board();
    init_knight_table();
    chess_board.print_board();
    init_magic_tables();
    init_king_mask();
    init_all_pinned_tables();
    init_all_atack_tables();

    move_stack move_stack;

    // Anzahl der Wiederholungen
    const int repetitions = 66000000; 

    // Zeitmessung starten
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repetitions; ++i) {
        move_stack.clear();
        find_all_moves(&move_stack, &chess_board);
    }

    // Zeitmessung beenden
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Alle Züge wurden " << repetitions << " mal generiert." << std::endl;
    std::cout << "Dauer: " << duration.count() << " Millisekunden." << std::endl;

    std::cout << "\nBeispielausgabe der letzten Generierung:" << std::endl;
    move_stack.print_moves();

    // --------- SFML Fenster erstellen ---------
    sf::RenderWindow window(sf::VideoMode(8 * TILE_SIZE, 8 * TILE_SIZE), "Schach GUI");

    // Texturen für weiße Figuren
    sf::Texture white_pawn_tex, white_rook_tex, white_knight_tex, white_bishop_tex, white_queen_tex, white_king_tex;

    // Texturen für schwarze Figuren
    sf::Texture black_pawn_tex, black_rook_tex, black_knight_tex, black_bishop_tex, black_queen_tex, black_king_tex;

    // Lade Figuren Texturen (hier nur Beispiel für weißen Bauern)
    auto loadTexture = [](sf::Texture& tex, const std::string& filename) {
        if (!tex.loadFromFile(filename)) {
            std::cerr << "Fehler beim Laden der Textur " << filename << std::endl;
            exit(-1);  // Programm beenden bei Fehler
            }
    };

    loadTexture(white_pawn_tex, "./chess_images/white_pawn.png");
    loadTexture(white_rook_tex, "./chess_images/white_rook.png");
    loadTexture(white_knight_tex, "./chess_images/white_knight.png");
    loadTexture(white_bishop_tex, "./chess_images/white_bishop.png");
    loadTexture(white_queen_tex, "./chess_images/white_queen.png");
    loadTexture(white_king_tex, "./chess_images/white_king.png");

    loadTexture(black_pawn_tex, "./chess_images/black_pawn.png");
    loadTexture(black_rook_tex, "./chess_images/black_rook.png");
    loadTexture(black_knight_tex, "./chess_images/black_knight.png");
    loadTexture(black_bishop_tex, "./chess_images/black_bishop.png");
    loadTexture(black_queen_tex, "./chess_images/black_queen.png");
    loadTexture(black_king_tex, "./chess_images/black_king.png");
    // Sprites erzeugen
    sf::Sprite white_pawn_sprite(white_pawn_tex);
    sf::Sprite white_rook_sprite(white_rook_tex);
    sf::Sprite white_knight_sprite(white_knight_tex);
    sf::Sprite white_bishop_sprite(white_bishop_tex);
    sf::Sprite white_queen_sprite(white_queen_tex);
    sf::Sprite white_king_sprite(white_king_tex);

    sf::Sprite black_pawn_sprite(black_pawn_tex);
    sf::Sprite black_rook_sprite(black_rook_tex);
    sf::Sprite black_knight_sprite(black_knight_tex);
    sf::Sprite black_bishop_sprite(black_bishop_tex);
    sf::Sprite black_queen_sprite(black_queen_tex);
    sf::Sprite black_king_sprite(black_king_tex);

    // Weitere Figuren laden analog...

    // Main Loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();

        // Schachbrett zeichnen
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                sf::RectangleShape square(sf::Vector2f(TILE_SIZE, TILE_SIZE));
                square.setPosition(col * TILE_SIZE, row * TILE_SIZE);

                if ((row + col) % 2 == 0)
                    square.setFillColor(sf::Color(240, 217, 181)); // hell
                else
                    square.setFillColor(sf::Color(181, 136, 99));  // dunkel

                window.draw(square);
            }
        }
        int offset_x = 10;
        int offset_y = 10;
        // Beispiel: Zeichne weißen Bauern auf den Feldern entsprechend dem Bitboard
        for (int bit = 0; bit < 64; ++bit) {
            int row = 7 - (bit / 8);
            int col = bit % 8;
            sf::Vector2f pos(col * TILE_SIZE + offset_x, row * TILE_SIZE + offset_y);

            // Weiße Figuren zeichnen
            if (chess_board.white.pawns & (1ULL << bit)) {
                white_pawn_sprite.setPosition(pos);
                window.draw(white_pawn_sprite);
            }
            if (chess_board.white.rooks & (1ULL << bit)) {
                white_rook_sprite.setPosition(pos);
                window.draw(white_rook_sprite);
            }
            if (chess_board.white.knights & (1ULL << bit)) {
                white_knight_sprite.setPosition(pos);
                window.draw(white_knight_sprite);
            }
            if (chess_board.white.bishop & (1ULL << bit)) {
                white_bishop_sprite.setPosition(pos);
                window.draw(white_bishop_sprite);
            }
            if (chess_board.white.queen & (1ULL << bit)) {
                white_queen_sprite.setPosition(pos);
                window.draw(white_queen_sprite);
            }
            if (chess_board.white.king & (1ULL << bit)) {
                white_king_sprite.setPosition(pos);
                window.draw(white_king_sprite);
            }

            // Schwarze Figuren zeichnen
            if (chess_board.black.pawns & (1ULL << bit)) {
                black_pawn_sprite.setPosition(pos);
                window.draw(black_pawn_sprite);
            }
            if (chess_board.black.rooks & (1ULL << bit)) {
                black_rook_sprite.setPosition(pos);
                window.draw(black_rook_sprite);
            }
            if (chess_board.black.knights & (1ULL << bit)) {
                black_knight_sprite.setPosition(pos);
                window.draw(black_knight_sprite);
            }
            if (chess_board.black.bishop & (1ULL << bit)) {
                black_bishop_sprite.setPosition(pos);
                window.draw(black_bishop_sprite);
            }
            if (chess_board.black.queen & (1ULL << bit)) {
                black_queen_sprite.setPosition(pos);
                window.draw(black_queen_sprite);
            }
            if (chess_board.black.king & (1ULL << bit)) {
                black_king_sprite.setPosition(pos);
                window.draw(black_king_sprite);
            }
        }

        // Analog für andere Figuren (schwarz & weiß) hinzufügen

        window.display();
    }

    return 0;
}
