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

# 🔹 1. Root-Zug: h2 -> h4
root_move = chess.Move.from_uci("h2h4")
assert root_move in board.legal_moves

board.push(root_move)

print("After h2h4:\n")

total = 0

# 🔹 2. Alle Folgezüge, perft(1)
for move in board.legal_moves:
    board.push(move)
    nodes = perft(board, 1)
    board.pop()

    print(f"{move.uci()}: {nodes}")
    total += nodes

print("\nTotal:", total)
