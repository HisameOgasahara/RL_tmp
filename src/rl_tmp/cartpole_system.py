from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class CartPoleConfig:
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5
    max_force: float = 10.0


def cartpole_field_numpy(state: np.ndarray, force: float, config: CartPoleConfig) -> np.ndarray:
    x, theta, x_dot, theta_dot = state
    total_mass = config.cart_mass + config.pole_mass
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    temp = (force + config.pole_mass * config.half_length * theta_dot**2 * sin_theta) / total_mass
    theta_ddot = (
        config.gravity * sin_theta - cos_theta * temp
    ) / (
        config.half_length
        * (4.0 / 3.0 - config.pole_mass * cos_theta**2 / total_mass)
    )
    x_ddot = temp - (
        config.pole_mass * config.half_length * theta_ddot * cos_theta / total_mass
    )
    return np.array([x_dot, theta_dot, x_ddot, theta_ddot], dtype=np.float64)


def cartpole_field_torch(state: torch.Tensor, force: torch.Tensor, config: CartPoleConfig) -> torch.Tensor:
    x, theta, x_dot, theta_dot = state.unbind(dim=-1)
    total_mass = config.cart_mass + config.pole_mass
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    force = force.squeeze(-1) if force.ndim == state.ndim else force
    temp = (
        force + config.pole_mass * config.half_length * theta_dot.square() * sin_theta
    ) / total_mass
    theta_ddot = (
        config.gravity * sin_theta - cos_theta * temp
    ) / (
        config.half_length
        * (4.0 / 3.0 - config.pole_mass * cos_theta.square() / total_mass)
    )
    x_ddot = temp - (
        config.pole_mass * config.half_length * theta_ddot * cos_theta / total_mass
    )
    return torch.stack((x_dot, theta_dot, x_ddot, theta_ddot), dim=-1)


def rk4_step_torch(state: torch.Tensor, force: torch.Tensor, dt: float, config: CartPoleConfig) -> torch.Tensor:
    k1 = cartpole_field_torch(state, force, config)
    k2 = cartpole_field_torch(state + 0.5 * dt * k1, force, config)
    k3 = cartpole_field_torch(state + 0.5 * dt * k2, force, config)
    k4 = cartpole_field_torch(state + dt * k3, force, config)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
