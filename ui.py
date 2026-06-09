from __future__ import annotations

from pathlib import Path

from agent import DQNAgent
from game import EMPTY, O, X, check_win, encode_state, free_spaces, print_board


def human_move(board: list[int]) -> int:
    while True:
        raw = input("Choose a space (1-9): ").strip()
        if not raw.isdigit():
            print("Enter a number from 1 to 9.")
            continue
        move = int(raw) - 1
        if move not in range(9):
            print("Move must be between 1 and 9.")
            continue
        if board[move] != EMPTY:
            print("That space is taken.")
            continue
        return move


def play_human_vs_agent(save_path: Path) -> None:
    agent = DQNAgent()
    if save_path.exists():
        agent.load(save_path)
    else:
        print(f"No saved model found at {save_path}. The agent will play untrained.")

    human_player = X if input("Play as X and go first? [Y/n]: ").strip().lower() not in {"n", "no"} else O
    current_player = X
    board = [EMPTY] * 9

    while True:
        print_board(board)
        if current_player == human_player:
            move = human_move(board)
        else:
            move = agent.select_action(encode_state(board, current_player), free_spaces(board), explore=False)
            print(f"Agent chooses {move + 1}")

        board[move] = current_player
        winner = check_win(board)
        if winner != EMPTY or not free_spaces(board):
            print_board(board)
            if winner == EMPTY:
                print("Draw.")
            elif winner == human_player:
                print("You win.")
            else:
                print("Agent wins.")
            return

        current_player = -current_player
