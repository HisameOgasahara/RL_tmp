import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .dynamics import (
    TwoLinkConfig,
    forward_kinematics,
    rk4_step,
)


class TwoLinkArmEnv(gym.Env):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        config: TwoLinkConfig | None = None,
        dt: float = 0.04,
        horizon: int = 150,
        target=(1.25, 1.0),
        initial_state=(0.0, 0.0, 0.0, 0.0),
        random_reset: bool = True,
    ):
        super().__init__()

        self.config = (
            config
            if config is not None
            else TwoLinkConfig()
        )

        self.dt = float(dt)
        self.horizon = int(horizon)
        self.target = np.asarray(
            target,
            dtype=np.float64,
        )

        self.initial_state = np.asarray(
            initial_state,
            dtype=np.float64,
        )

        self.random_reset = bool(
            random_reset
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.full(
                10,
                -np.inf,
                dtype=np.float32,
            ),
            high=np.full(
                10,
                np.inf,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.state = self.initial_state.copy()
        self.step_count = 0

    def _observation(self):
        q1, q2, dq1, dq2 = self.state

        return np.array(
            [
                np.cos(q1),
                np.sin(q1),
                np.cos(q2),
                np.sin(q2),
                dq1,
                dq2,
                self.target[0],
                self.target[1],
                self.state[0],
                self.state[1],
            ],
            dtype=np.float32,
        )

    def _distance(self):
        end_effector = forward_kinematics(
            self.state[:2],
            self.config,
        )

        return float(
            np.linalg.norm(
                end_effector
                - self.target
            )
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.state = self.initial_state.copy()

        if self.random_reset:
            self.state[:2] += self.np_random.uniform(
                low=-0.15,
                high=0.15,
                size=2,
            )

            self.state[2:] += self.np_random.uniform(
                low=-0.05,
                high=0.05,
                size=2,
            )

        self.step_count = 0

        return (
            self._observation(),
            {
                "state": self.state.copy(),
                "distance": self._distance(),
            },
        )

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float64,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        torque = (
            action
            * self.config.max_torque
        )

        integration_dt = (
            self.dt
            / 4.0
        )

        for _ in range(4):
            self.state = rk4_step(
                state=self.state,
                torque=torque,
                dt=integration_dt,
                config=self.config,
            )

        self.step_count += 1

        distance = self._distance()

        reward = (
            -4.0 * distance * distance
            -0.002 * float(
                np.dot(
                    torque,
                    torque,
                )
            )
            -0.01 * float(
                np.dot(
                    self.state[2:],
                    self.state[2:],
                )
            )
        )

        reached = (
            distance < 0.08
            and np.linalg.norm(
                self.state[2:]
            ) < 0.25
        )

        if reached:
            reward += 20.0

        terminated = bool(reached)
        truncated = bool(
            self.step_count
            >= self.horizon
        )

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {
                "state": self.state.copy(),
                "torque": torque.copy(),
                "distance": distance,
                "end_effector": forward_kinematics(
                    self.state[:2],
                    self.config,
                ),
            },
        )
