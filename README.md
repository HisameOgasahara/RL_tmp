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
- Gaussian Fokker-Planck PINN (mean/covariance moment residual)
- Neural SDE
- Reinforcement Learning (REINFORCE) under stochastic dynamics

SDE의 개별 sample path는 Wiener noise 때문에 일반적인 의미에서 미분 가능하지 않으므로, ODE PINN처럼 path 자체에 미분방정식 residual을 직접 걸기 어렵습니다. 그래서 PINN에서는 SDE가 유도하는 결정론적 Fokker-Planck equation을 사용합니다. 현재 예제는 선형 SDE + Gaussian 초기분포이므로 density가 계속 Gaussian이라는 구조를 이용해 mean/covariance moment equation을 PINN residual로 학습하며, density 정규화는 모델 구조로 보장합니다.

```text
dq = v dt
dv = (-k q - c v) dt + sigma dW
```

RL에서는 drift에 연속 제어 입력 `u`를 추가합니다.

```text
dv = (-k q - c v + u) dt + sigma dW
```

Colab: `notebooks/sde_methods_colab.ipynb`

## Nonlinear Cart-Pole Control

[![Open Cart-Pole In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HisameOgasahara/RL_tmp/blob/main/notebooks/cartpole_control_colab.ipynb)

비선형 Cart-Pole을 대상으로 다음을 같은 노트북에서 비교합니다.

- SciPy `solve_ivp(method="DOP853")` high-accuracy reference
- continuous-time LQR baseline
- learned nonlinear vector field / Neural ODE rollout
- PINN for the nonlinear LQR closed-loop ODE
- goal-reaching RL policy search (Cross-Entropy Method)

검증된 기본 실행에서는 초기 상태 `[x, theta, x_dot, theta_dot] = [0, 0.2, 0, 0]`, 목표 상태 `[0, 0, 0, 0]`에서 LQR의 5초 후 goal distance가 약 `0.0193`이었습니다. RL policy search도 학습한 정책을 실제 비선형 dynamics에 5초 rollout했을 때 final state distance to goal이 약 `0.0449`로 줄어 `goal_tolerance=0.1` 안에 들어왔습니다. Neural ODE의 1초 trajectory MSE는 약 `5.18e-3`, PINN의 1초 closed-loop trajectory MSE는 약 `3.54e-3`였습니다.

Colab: `notebooks/cartpole_control_colab.ipynb`

핵심 구현은 `src/rl_tmp/*.py`에 있고, Colab 노트북은 의존성 설치, git clone, 사용자 입력, 학습 실행, 결과 출력만 담당합니다.
