// gui.cpp
#include "gui.h"
#include <iostream>

// Konstanten
const int TILE_SIZE = 80;
const int OFFSET_X = 10;
const int OFFSET_Y = 10;

// Globale GUI-Objekte
sf::RenderWindow window;
sf::Texture textures[12]; // 0–5 = weiß, 6–11 = schwarz
sf::Sprite sprites[12];

enum PieceIndex {
    WP, WR, WN, WB, WQ, WK,
    BP, BR, BN, BB, BQ, BK
};

void load_texture(sf::Texture& tex, const std::string& path) {
    if (!tex.loadFromFile(path)) {
        std::cerr << "Fehler beim Laden: " << path << std::endl;
        std::exit(1);
    }
}

void init_gui() {
    window.create(sf::VideoMode(8 * TILE_SIZE, 8 * TILE_SIZE), "Schach GUI");

    // Texturen laden
    load_texture(textures[WP], "./chess_images/white_pawn.png");
    load_texture(textures[WR], "./chess_images/white_rook.png");
    load_texture(textures[WN], "./chess_images/white_knight.png");
    load_texture(textures[WB], "./chess_images/white_bishop.png");
    load_texture(textures[WQ], "./chess_images/white_queen.png");
    load_texture(textures[WK], "./chess_images/white_king.png");

    load_texture(textures[BP], "./chess_images/black_pawn.png");
    load_texture(textures[BR], "./chess_images/black_rook.png");
    load_texture(textures[BN], "./chess_images/black_knight.png");
    load_texture(textures[BB], "./chess_images/black_bishop.png");
    load_texture(textures[BQ], "./chess_images/black_queen.png");
    load_texture(textures[BK], "./chess_images/black_king.png");

    // Sprites setzen
    for (int i = 0; i < 12; ++i) {
        sprites[i].setTexture(textures[i]);
    }
}

bool update_gui(const chess_board& board) {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
            return false;
        }
    }

    window.clear();

    // Brett zeichnen
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            sf::RectangleShape square(sf::Vector2f(TILE_SIZE, TILE_SIZE));
            square.setPosition(col * TILE_SIZE, row * TILE_SIZE);
            if ((row + col) % 2 == 0)
                square.setFillColor(sf::Color(240, 217, 181));
            else
                square.setFillColor(sf::Color(181, 136, 99));
            window.draw(square);
        }
    }

    // Figuren zeichnen
    auto draw_pieces = [&](const one_side& side, bool white) {
        int offset = white ? 0 : 6;
        for (int bit = 0; bit < 64; ++bit) {
            int row = 7 - (bit / 8);
            int col = bit % 8;
            sf::Vector2f pos(col * TILE_SIZE + OFFSET_X, row * TILE_SIZE + OFFSET_Y);

            if (side.pawns & (1ULL << bit)) {
                sprites[offset + 0].setPosition(pos);
                window.draw(sprites[offset + 0]);
            }
            if (side.rooks & (1ULL << bit)) {
                sprites[offset + 1].setPosition(pos);
                window.draw(sprites[offset + 1]);
            }
            if (side.knights & (1ULL << bit)) {
                sprites[offset + 2].setPosition(pos);
                window.draw(sprites[offset + 2]);
            }
            if (side.bishop & (1ULL << bit)) {
                sprites[offset + 3].setPosition(pos);
                window.draw(sprites[offset + 3]);
            }
            if (side.queen & (1ULL << bit)) {
                sprites[offset + 4].setPosition(pos);
                window.draw(sprites[offset + 4]);
            }
            if (side.king & (1ULL << bit)) {
                sprites[offset + 5].setPosition(pos);
                window.draw(sprites[offset + 5]);
            }
        }
    };

    draw_pieces(board.white, true);
    draw_pieces(board.black, false);

    window.display();
    return true;
}
