// gui.cpp
#include "gui.h"
#include <iostream>
#include <set>

#ifndef ASSET_DIR_PATH
#define ASSET_DIR_PATH "./chess_gui/chess_images/"
#endif

// Constants
const int OFFSET_X = 10;
const int OFFSET_Y = 10;
const int TILE_SIZE = 80;
const int BUTTON_AREA_HEIGHT = 50;  // Space reserved for the button on top

bool legal_moves_only = true;

// Global GUI objects
sf::RenderWindow window;
sf::Texture textures[12]; // 0–5 = white, 6–11 = black
sf::Sprite sprites[12];

using std::filesystem::path;
constexpr const char* ASSET_DIR = ASSET_DIR_PATH;



// Global or passed to update_gui (your call)
std::set<int> highlighted_squares;

void highlight_moves(const move_stack& moves) {
    highlighted_squares.clear();
    for (int i = 0; i < moves.move_counter; i += 4) {
        int to = __builtin_ctzll(moves.moves[i + 1]);
        if (to >= 0 && to < 64) {
            highlighted_squares.insert(to);
        }
    }
}

enum PieceIndex {
    WP, WR, WN, WB, WQ, WK,
    BP, BR, BN, BB, BQ, BK
};

void load_texture(sf::Texture& tex, const std::string& path) {
    if (!tex.loadFromFile(path)) {
        std::cerr << "Error loading: " << path << std::endl;
        std::exit(1);
    }
}

void init_gui() {
    window.create(sf::VideoMode(8 * TILE_SIZE, 8 * TILE_SIZE + BUTTON_AREA_HEIGHT), "Chess GUI");

    // Draw button
    // Load font
    sf::Font font;
    if (!font.loadFromFile(std::string(ASSET_DIR) + "arial.TTF")) {
        std::cerr << "Error loading font!" << std::endl;
    }


    // Load textures
    load_texture(textures[WP], std::string(ASSET_DIR) + "white_pawn.png");
    load_texture(textures[WR], std::string(ASSET_DIR) + "white_rook.png");
    load_texture(textures[WN], std::string(ASSET_DIR) + "white_knight.png");
    load_texture(textures[WB], std::string(ASSET_DIR) + "white_bishop.png");
    load_texture(textures[WQ], std::string(ASSET_DIR) + "white_queen.png");
    load_texture(textures[WK], std::string(ASSET_DIR) + "white_king.png");

    load_texture(textures[BP], std::string(ASSET_DIR) + "black_pawn.png");
    load_texture(textures[BR], std::string(ASSET_DIR) + "black_rook.png");
    load_texture(textures[BN], std::string(ASSET_DIR) + "black_knight.png");
    load_texture(textures[BB], std::string(ASSET_DIR) + "black_bishop.png");
    load_texture(textures[BQ], std::string(ASSET_DIR) + "black_queen.png");
    load_texture(textures[BK], std::string(ASSET_DIR) + "black_king.png");


    // Assign textures to sprites
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

        // Handle mouse click event
        if (event.type == sf::Event::MouseButtonPressed) {
            if (event.mouseButton.button == sf::Mouse::Left) {
                sf::Vector2f click_pos(event.mouseButton.x, event.mouseButton.y);
                sf::FloatRect button_bounds(10, 5, 200, 40); // Button is located e.g. at (10,5) in the button area
                if (button_bounds.contains(click_pos)) {
                    legal_moves_only = !legal_moves_only;
                    std::cout << "legal_moves_only is now: "
                              << (legal_moves_only ? "true" : "false") << "\n";
                }
            }
        }
    }

    window.clear();

    // Draw button (before drawing the chessboard)
    sf::RectangleShape button(sf::Vector2f(200, 40));
    button.setPosition(10, 5);  // In the top button area
    button.setFillColor(legal_moves_only ? sf::Color(100, 200, 100)
                                         : sf::Color(200, 100, 100));
    window.draw(button);

    // Button text
    static sf::Font font;
    static bool fontLoaded = false;
    if (!fontLoaded) {
        if (!fontLoaded) {
            if (!font.loadFromFile(std::string(ASSET_DIR) + "arial.TTF")) {
                std::cerr << "Error loading font!" << std::endl;
            } else {
                fontLoaded = true;
            }
        }
    }

    sf::Text button_text;
    button_text.setFont(font);
    button_text.setCharacterSize(18);
    button_text.setFillColor(sf::Color::Black);
    button_text.setString(legal_moves_only ? "only legal moves" : "all moves");
    button_text.setPosition(20, 10);
    window.draw(button_text);

    // Draw chessboard (shifted down due to button area)
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            int bit_index = (7 - row) * 8 + col; // Bitboard index = reversed row
            bool is_highlighted = highlighted_squares.count(bit_index) > 0;

            sf::RectangleShape square(sf::Vector2f(TILE_SIZE, TILE_SIZE));
            square.setPosition(col * TILE_SIZE, row * TILE_SIZE + BUTTON_AREA_HEIGHT);

            if ((row + col) % 2 == 0)
                square.setFillColor(is_highlighted ? sf::Color(200, 255, 200)
                                                   : sf::Color(240, 217, 181));
            else
                square.setFillColor(is_highlighted ? sf::Color(100, 180, 100)
                                                   : sf::Color(181, 136, 99));

            window.draw(square);
        }
    }

    // Draw pieces (also shifted by button height)
    auto draw_pieces = [&](const one_side& side, bool white) {
        int offset = white ? 0 : 6;
        for (int bit = 0; bit < 64; ++bit) {
            int row = 7 - (bit / 8);
            int col = bit % 8;
            sf::Vector2f pos(col * TILE_SIZE + OFFSET_X,
                             row * TILE_SIZE + OFFSET_Y + BUTTON_AREA_HEIGHT);

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
