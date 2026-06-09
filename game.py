from __future__ import annotations

EMPTY = 0
X = 1
O = -1


def check_win(board: list[int]) -> int:
    lines = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )
    for a, b, c in lines:
        if board[a] == board[b] == board[c] != EMPTY:
            return board[a]
    return EMPTY


def free_spaces(board: list[int]) -> list[int]:
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def encode_state(board: list[int], player: int) -> list[float]:
    return [float(cell * player) for cell in board]


def terminal_reward(winner: int, player: int) -> float:
    if winner == player:
        return 1.0
    if winner == -player:
        return -1.0
    return 0.0


def symbol(value: int) -> str:
    if value == X:
        return "X"
    if value == O:
        return "O"
    return " "


def print_board(board: list[int]) -> None:
    rows = []
    for start in (0, 3, 6):
        rows.append(" | ".join(symbol(board[start + i]) for i in range(3)))
    print("\n---------")
    print("\n---------\n".join(rows))
    print()
