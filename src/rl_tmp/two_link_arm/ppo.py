from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .env import TwoLinkArmEnv


class PPOCheckpointCallback(
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
            ] = _snapshot_policy(self.model.policy)

        if (
            "middle"
            not in self.snapshots
            and progress >= 0.5
        ):
            self.snapshots[
                "middle"
            ] = _snapshot_policy(self.model.policy)

        return True


def train_ppo(
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

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        device=device,
    )

    callback = PPOCheckpointCallback(
        total_timesteps=total_timesteps,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    callback.snapshots[
        "final"
    ] = _snapshot_policy(model.policy)

    return (
        model,
        callback.snapshots,
    )
