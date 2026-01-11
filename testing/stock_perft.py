import chess

def perft(board, depth):
    if depth == 0:
        return 1

    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


board = chess.Board()

# 1️⃣ b2 -> b3
move1 = chess.Move.from_uci("b2b3")
assert move1 in board.legal_moves
board.push(move1)

# 2️⃣ a7 -> a6
move2 = chess.Move.from_uci("a7a6")
assert move2 in board.legal_moves
board.push(move2)

print("After b2b3 a7a6:\n")

total = 0

# 3️⃣ Alle Folgezüge, perft(1)
for move in board.legal_moves:
    board.push(move)
    nodes = perft(board, 1)
    board.pop()

    print(f"{move.uci()}: {nodes}")
    total += nodes

print("\nTotal:", total)
