# RL_tmp

강화학습과 동역학을 작은 물리계에서 비교하는 실습 저장소입니다.

## Colab notebooks

### ODE

[![Open ODE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/ode_methods_colab.ipynb)

- Notebook: `notebooks/ode_methods_colab.ipynb`
- Code: `src/rl_tmp/ode_methods/`
- Details: [docs/ode_methods.md](docs/ode_methods.md)

### SDE

[![Open SDE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/sde_methods_colab.ipynb)

- Notebook: `notebooks/sde_methods_colab.ipynb`
- Code: `src/rl_tmp/sde_methods/`
- Details: [docs/sde_methods.md](docs/sde_methods.md)

### Nonlinear Cart-Pole control

[![Open Cart-Pole In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/cartpole_control_colab.ipynb)

- Notebook: `notebooks/cartpole_control_colab.ipynb`
- Code: `src/rl_tmp/cartpole_control/`
- Details: [docs/cartpole_control.md](docs/cartpole_control.md)

### PyBullet 2-link arm — CEM

[![Open 2-Link CEM In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/two_link_cem_colab.ipynb)

### PyBullet 2-link arm — PPO

[![Open 2-Link PPO In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/two_link_ppo_colab.ipynb)

### PyBullet 2-link arm — SAC

[![Open 2-Link SAC In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/two_link_sac_colab.ipynb)

세 알고리즘은 같은 Euler-Lagrange 2-link arm 환경을 각각 독립적으로 학습합니다. PyBullet 로봇팔 애니메이션과 상태공간 trajectory를 같은 rollout에서 동기화해 비교합니다.

- Code: `src/rl_tmp/two_link_arm/`
- Details: [docs/two_link_arm.md](docs/two_link_arm.md)
