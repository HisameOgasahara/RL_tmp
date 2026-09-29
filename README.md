# RL_tmp

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

핵심 구현은 `src/rl_tmp/*.py`에 있고, Colab 노트북은 의존성 설치, git clone, 사용자 입력, 학습 실행, 결과 출력만 담당합니다.

Colab: `notebooks/ode_methods_colab.ipynb`
