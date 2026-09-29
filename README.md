# RL_tmp

## ODE

[![Open ODE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/ode_methods_colab.ipynb)

하나의 2차원 ODE를 기준으로 다음 방법을 비교하는 학습용 프로젝트입니다.

- 수치 ODE solver
- PINN
- Neural ODE
- Reinforcement Learning (REINFORCE)

공통 시스템은 감쇠 조화진동자입니다.

```text
state x = (q, v)
dq/dt = v
dv/dt = -k q - c v
```

RL에서는 같은 시스템에 연속 제어 입력 `u`만 추가합니다.

```text
dv/dt = -k q - c v + u
```

Colab: `notebooks/ode_methods_colab.ipynb`

## SDE

[![Open SDE In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/sde_methods_colab.ipynb)

같은 감쇠 조화진동자에 velocity noise를 넣은 SDE를 기준으로 다음 방법을 비교합니다.

- Euler-Maruyama numerical solver
- Fokker-Planck PINN
- Neural SDE
- Reinforcement Learning (REINFORCE) under stochastic dynamics

```text
dq = v dt
dv = (-k q - c v) dt + sigma dW
```

RL에서는 drift에 연속 제어 입력 `u`를 추가합니다.

```text
dv = (-k q - c v + u) dt + sigma dW
```

Colab: `notebooks/sde_methods_colab.ipynb`

핵심 구현은 `src/rl_tmp/*.py`에 있고, Colab 노트북은 의존성 설치, git clone, 사용자 입력, 학습 실행, 결과 출력만 담당합니다.
