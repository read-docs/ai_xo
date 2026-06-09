from __future__ import annotations

import argparse
from pathlib import Path

from train import train_agent
from ui import play_human_vs_agent

MODEL_PATH = Path("tic_tac_toe_dqn.pt")
LOG_PATH = Path("training_log.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or play Tic Tac Toe DQN")
    parser.add_argument("--mode", choices=("train", "play"), default="train")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--save-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "play":
        play_human_vs_agent(args.save_path)
    else:
        train_agent(args.episodes, args.eval_every, args.eval_games, args.save_path, args.log_path)


if __name__ == "__main__":
    main()
