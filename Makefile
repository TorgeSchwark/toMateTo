# Compiler
CXX = g++
CXXFLAGS = -Wall -std=c++17 -O3

# Zielprogramm
TARGET = chess

# Quell- und Objektdateien
SRC = main.cpp ./move_generation/chess_board.cpp ./generation/knight_tables.cpp  ./generation/magic_gen.cpp ./move_generation/move_stack.cpp
OBJ = $(SRC:.cpp=.o)

# Standardziel
all: $(TARGET)

# Linken
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Kompilieren
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Aufräumen
clean:
	rm -f *.o $(TARGET)
