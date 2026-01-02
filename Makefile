# Compiler
CXX = g++
CXXFLAGS = -Wall -std=c++17 -O3

# SFML libraries (adjust path if needed)
SFML_LIBS = -lsfml-graphics -lsfml-window -lsfml-system

# Target program
TARGET = chess

# Source and object files
SRC = main.cpp ./chess_gui/gui.cpp ./move_generation/chess_board.cpp ./generation/knight_tables.cpp ./generation/king_tables.cpp  ./generation/magic_gen.cpp ./move_generation/move_stack.cpp 
OBJ = $(SRC:.cpp=.o)

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(SFML_LIBS)

# Compiling
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Cleanup
clean:
	rm -f *.o $(TARGET)
