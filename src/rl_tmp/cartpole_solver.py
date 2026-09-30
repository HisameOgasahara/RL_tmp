from collections.abc import Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from .cartpole_system import CartPoleConfig, cartpole_field_numpy

ControlLaw = Callable[[np.ndarray], float]

def solve_cartpole_reference(initial_state, times, config, control_law=None):
    initial_state = np.asarray(initial_state, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    def field(_t, state):
        force = 0.0 if control_law is None else float(control_law(state))
        return cartpole_field_numpy(state, force, config)
    result = solve_ivp(
        field,
        (float(times[0]), float(times[-1])),
        initial_state,
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y.T


def linearize_cartpole(config, epsilon=1e-6):
    equilibrium = np.zeros(4, dtype=np.float64)
    A = np.zeros((4, 4), dtype=np.float64)
    for index in range(4):
        delta = np.zeros(4, dtype=np.float64)
        delta[index] = epsilon
        A[:, index] = (
            cartpole_field_numpy(equilibrium + delta, 0.0, config)
            - cartpole_field_numpy(equilibrium - delta, 0.0, config)
        ) / (2.0 * epsilon)
    B = (
        cartpole_field_numpy(equilibrium, epsilon, config)
        - cartpole_field_numpy(equilibrium, -epsilon, config)
    ).reshape(-1, 1) / (2.0 * epsilon)
    return A, B


def make_lqr_controller(config, q_diagonal=(1.0, 12.0, 1.0, 2.0), r=0.15):
    A, B = linearize_cartpole(config)
    Q = np.diag(q_diagonal)
    R = np.array([[r]], dtype=np.float64)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    def control(state):
        force = -(K @ np.asarray(state, dtype=np.float64))[0]
        return float(np.clip(force, -config.max_force, config.max_force))
    return control, K, A, B
