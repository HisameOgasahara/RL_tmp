from dataclasses import dataclass

import numpy as np

from .env import TwoLinkArmEnv


@dataclass
class CEMResult:
    mean_parameters: np.ndarray
    history: dict[str, list[float]]
    checkpoints: dict[str, np.ndarray]


class CEMPolicy:
    def __init__(
        self,
        parameters,
        hidden_size: int = 16,
    ):
        self.parameters = np.asarray(
            parameters,
            dtype=np.float64,
        )

        self.hidden_size = int(
            hidden_size
        )

    @staticmethod
    def parameter_count(
        observation_size: int = 10,
        hidden_size: int = 16,
        action_size: int = 2,
    ):
        return (
            observation_size * hidden_size
            + hidden_size
            + hidden_size * action_size
            + action_size
        )

    def __call__(self, observation):
        observation = np.asarray(
            observation,
            dtype=np.float64,
        )

        offset = 0

        size = 10 * self.hidden_size
        w1 = self.parameters[
            offset : offset + size
        ].reshape(
            10,
            self.hidden_size,
        )

        offset += size

        b1 = self.parameters[
            offset : offset + self.hidden_size
        ]

        offset += self.hidden_size

        size = self.hidden_size * 2
        w2 = self.parameters[
            offset : offset + size
        ].reshape(
            self.hidden_size,
            2,
        )

        offset += size

        b2 = self.parameters[
            offset : offset + 2
        ]

        hidden = np.tanh(
            observation @ w1 + b1
        )

        return np.tanh(
            hidden @ w2 + b2
        ).astype(
            np.float32
        )


def evaluate_parameters(
    parameters,
    env_kwargs,
    seed: int,
):
    env = TwoLinkArmEnv(
        **env_kwargs
    )

    observation, _ = env.reset(
        seed=seed
    )

    policy = CEMPolicy(
        parameters
    )

    total_reward = 0.0

    while True:
        action = policy(
            observation
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(
            action
        )

        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    return float(
        total_reward
    )


def train_cem(
    env_kwargs=None,
    iterations: int = 35,
    population_size: int = 192,
    elite_fraction: float = 0.12,
    initial_std: float = 0.7,
    minimum_std: float = 0.03,
    seed: int = 0,
):
    if env_kwargs is None:
        env_kwargs = {}

    rng = np.random.default_rng(
        seed
    )

    parameter_count = CEMPolicy.parameter_count()

    mean = np.zeros(
        parameter_count,
        dtype=np.float64,
    )

    std = np.full(
        parameter_count,
        initial_std,
        dtype=np.float64,
    )

    elite_count = max(
        1,
        int(
            round(
                population_size
                * elite_fraction
            )
        ),
    )

    history = {
        "elite_mean_reward": [],
        "best_reward": [],
    }

    checkpoints = {}

    for iteration in range(
        iterations
    ):
        candidates = (
            mean[None, :]
            + std[None, :]
            * rng.standard_normal(
                (
                    population_size,
                    parameter_count,
                )
            )
        )

        rewards = np.array(
            [
                evaluate_parameters(
                    candidate,
                    env_kwargs,
                    seed=seed + index,
                )
                for index, candidate
                in enumerate(candidates)
            ],
            dtype=np.float64,
        )

        elite_indices = np.argsort(
            rewards
        )[-elite_count:]

        elites = candidates[
            elite_indices
        ]

        mean = elites.mean(
            axis=0
        )

        std = np.maximum(
            elites.std(
                axis=0
            ),
            minimum_std,
        )

        elite_mean_reward = float(
            rewards[
                elite_indices
            ].mean()
        )

        best_reward = float(
            rewards.max()
        )

        history[
            "elite_mean_reward"
        ].append(
            elite_mean_reward
        )

        history[
            "best_reward"
        ].append(
            best_reward
        )

        if iteration == 0:
            checkpoints[
                "early"
            ] = mean.copy()

        if iteration == (
            iterations // 2
        ):
            checkpoints[
                "middle"
            ] = mean.copy()

    checkpoints[
        "final"
    ] = mean.copy()

    return CEMResult(
        mean_parameters=mean,
        history=history,
        checkpoints=checkpoints,
    )
