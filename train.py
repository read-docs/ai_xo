from __future__ import annotations

import csv
import random
from pathlib import Path

from agent import DQNAgent
from game import EMPTY, O, X, check_win, encode_state, free_spaces, terminal_reward


class TrainingLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._needs_header = not path.exists()

    def write_header(self) -> None:
        if self._needs_header:
            with self.path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "winner", "epsilon", "eval_wins", "eval_draws", "eval_losses"])
            self._needs_header = False

    def log(self, episode: int, winner: int, epsilon: float, wins: int, draws: int, losses: int) -> None:
        self.write_header()
        with self.path.open("a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode, winner, f"{epsilon:.6f}", wins, draws, losses])


def play_self_play_episode(agent: DQNAgent) -> int:
    board = [EMPTY] * 9
    pending: tuple[int, list[float], int] | None = None
    current_player = random.choice((X, O))

    while True:
        state = encode_state(board, current_player)
        action = agent.select_action(state, free_spaces(board), explore=True)
        board[action] = current_player

        winner = check_win(board)
        done = winner != EMPTY or not free_spaces(board)

        if pending is not None:
            prev_player, prev_state, prev_action = pending
            prev_reward = terminal_reward(winner, prev_player) if done else 0.0
            agent.remember(prev_state, prev_action, prev_reward, encode_state(board, prev_player), done)
            agent.optimize()
            pending = None

        if done:
            agent.remember(state, action, terminal_reward(winner, current_player), encode_state(board, current_player), True)
            agent.optimize()
            return winner

        pending = (current_player, state, action)
        current_player = -current_player


def play_vs_random(agent: DQNAgent, agent_player: int) -> int:
    board = [EMPTY] * 9
    current_player = X

    while True:
        if current_player == agent_player:
            action = agent.select_action(encode_state(board, current_player), free_spaces(board), explore=False)
        else:
            action = random.choice(free_spaces(board))

        board[action] = current_player
        winner = check_win(board)
        if winner != EMPTY or not free_spaces(board):
            return winner

        current_player = -current_player


def evaluate(agent: DQNAgent, games: int = 100) -> tuple[int, int, int]:
    wins = draws = losses = 0
    for game in range(games):
        agent_player = X if game % 2 == 0 else O
        winner = play_vs_random(agent, agent_player)
        if winner == 0:
            draws += 1
        elif winner == agent_player:
            wins += 1
        else:
            losses += 1
    return wins, draws, losses


def train_agent(episodes: int, eval_every: int, eval_games: int, save_path: Path, log_path: Path) -> None:
    agent = DQNAgent()
    if save_path.exists():
        agent.load(save_path)

    logger = TrainingLogger(log_path)

    for episode in range(1, episodes + 1):
        winner = play_self_play_episode(agent)
        agent.decay_epsilon()

        if episode % eval_every == 0:
            wins, draws, losses = evaluate(agent, eval_games)
            logger.log(episode, winner, agent.epsilon, wins, draws, losses)
            print(
                f"episode={episode} winner={winner} epsilon={agent.epsilon:.3f} "
                f"eval_wins={wins} eval_draws={draws} eval_losses={losses}"
            )
            agent.save(save_path)

    agent.save(save_path)
