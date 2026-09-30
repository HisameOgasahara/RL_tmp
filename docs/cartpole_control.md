# Nonlinear Cart-Pole control

상태와 제어입력은

$$
x(t)=
\begin{pmatrix}
p(t)\\
\theta(t)\\
\dot p(t)\\
\dot\theta(t)
\end{pmatrix}
\in\mathbb{R}^4,
\qquad
u(t)\in\mathbb{R}.
$$

$m_c>0$는 cart mass, $m_p>0$는 pole mass, $\ell>0$는 pole half-length, $g>0$는 중력가속도입니다.

$$
M=m_c+m_p
$$

$$
a(x,u)=\frac{u+m_p\ell\dot\theta^2\sin\theta}{M}
$$

현재 코드의 비선형 Cart-Pole dynamics는

$$
\dot p=\dot p,
\qquad
\dot\theta=\dot\theta
$$

$$
\ddot\theta=
\frac{
g\sin\theta-\cos\theta\,a(x,u)
}{
\ell\left(
\frac{4}{3}
-\frac{m_p\cos^2\theta}{M}
\right)
}
$$

$$
\ddot p=
a(x,u)
-
\frac{m_p\ell\ddot\theta\cos\theta}{M}.
$$

즉

$$
\dot x(t)=f(x(t),u(t)),
\qquad
f:\mathbb{R}^4\times\mathbb{R}\to\mathbb{R}^4.
$$

목표 상태는

$$
x_{\mathrm{goal}}=(0,0,0,0)
$$

입니다.

전체 vector field는 4차원이므로 한 평면에 직접 그릴 수 없습니다. 노트북에서는 $p=\dot p=0$으로 고정한 $(\theta,\dot\theta)$ 2차원 slice에서 uncontrolled / LQR / learned closed-loop vector field와 trajectory를 시각화합니다.

## Notebook

- Colab: `notebooks/cartpole_control_colab.ipynb`
- Core code: `src/rl_tmp/cartpole_control/`
