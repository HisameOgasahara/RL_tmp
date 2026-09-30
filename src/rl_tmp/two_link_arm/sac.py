from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .env import TwoLinkArmEnv


class SACCheckpointCallback(
    BaseCallback
):
    def __init__(
        self,
        total_timesteps: int,
    ):
        super().__init__()

        self.total_timesteps = int(
            total_timesteps
        )

        self.snapshots = {}

    def _on_step(self):
        progress = (
            self.num_timesteps
            / max(
                self.total_timesteps,
                1,
            )
        )

        if (
            "early"
            not in self.snapshots
            and progress >= 0.1
        ):
            self.snapshots[
                "early"
            ] = self.model.policy.state_dict()

        if (
            "middle"
            not in self.snapshots
            and progress >= 0.5
        ):
            self.snapshots[
                "middle"
            ] = self.model.policy.state_dict()

        return True


def train_sac(
    env_kwargs=None,
    total_timesteps: int = 80_000,
    seed: int = 0,
    device: str = "auto",
):
    if env_kwargs is None:
        env_kwargs = {}

    env = TwoLinkArmEnv(
        **env_kwargs
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        seed=seed,
        device=device,
    )

    callback = SACCheckpointCallback(
        total_timesteps=total_timesteps,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    callback.snapshots[
        "final"
    ] = model.policy.state_dict()

    return (
        model,
        callback.snapshots,
    )
