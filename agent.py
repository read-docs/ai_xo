from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: tuple[list[float], int, float, list[float], bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_interval: int = 500,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.optimize_steps = 0

    def _state_tensor(self, state: list[float]) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def select_action(self, state: list[float], legal_moves: list[int], explore: bool = True) -> int:
        if not legal_moves:
            raise ValueError("No legal moves available")

        if explore and random.random() < self.epsilon:
            return random.choice(legal_moves)

        with torch.no_grad():
            q_values = self.policy_net(self._state_tensor(state))
            illegal_moves = set(range(9)) - set(legal_moves)
            for move in illegal_moves:
                q_values[move] = -1e9
            return int(torch.argmax(q_values).item())

    def remember(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ) -> None:
        self.replay.push((state, action, reward, next_state, done))

    def optimize(self) -> None:
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.optimize_steps += 1
        if self.optimize_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "optimize_steps": self.optimize_steps,
            },
            path,
        )

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.optimize_steps = int(data.get("optimize_steps", 0))
