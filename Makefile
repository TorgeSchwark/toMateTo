# Compiler
CXX = g++
CXXFLAGS = -Wall -std=c++17 -O3

# SFML Libraries (ggf. Pfad anpassen)
SFML_LIBS = -lsfml-graphics -lsfml-window -lsfml-system

# Zielprogramm
TARGET = chess

# Quell- und Objektdateien
SRC = main.cpp ./chess_gui/gui.cpp ./move_generation/chess_board.cpp ./generation/knight_tables.cpp ./generation/king_tables.cpp  ./generation/magic_gen.cpp ./move_generation/move_stack.cpp 
OBJ = $(SRC:.cpp=.o)

# Standardziel
all: $(TARGET)

# Linken
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(SFML_LIBS)

# Kompilieren
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Aufräumen
clean:
	rm -f *.o $(TARGET)
