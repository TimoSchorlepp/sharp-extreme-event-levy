# Instantons for Lévy-driven SDEs via time-reparameterized optimal control

A minimal self-contained JAX script accompanying the paper

> J. Pan and T. Schorlepp, *Computing rare events in Lévy-driven systems via optimal control* (2026).

The script [`tr_instantons_2d.py`](tr_instantons_2d.py) computes, for a two-dimensional nonlinear SDE driven by four different noise types, the instanton (most likely path to a rare event), the rate function $I(z)$ and the prefactor $C(z)$ of the event probability. It uses the time-reparameterized (TR) optimal control formulation of the paper, which can also resolve instantons that jump.

## Setup

### Model

We consider the SDE (Sec. 5.1 of the paper)

$$
dX^\varepsilon_t = b(X^\varepsilon_t)dt + dZ^\varepsilon_t, \qquad
b(x,y) = (-x - xy, -4y + x^2)^\top, \qquad X_0^\varepsilon = (0,0),
$$

on $t \in [0, T]$ with $T = 1$, and ask for the probability of the rare event $X_T + 2 Y_T \geq z$ as $\varepsilon \to 0$.

The noise $Z^\varepsilon_t = \varepsilon Z_{t/\varepsilon}$ is a rescaled Lévy process: as $\varepsilon \to 0$, its jumps become small and frequent. Its two components are independent, with cumulant generating function (CGF)

$$
\Psi(p) = \psi(a_1 p_1) + \psi(a_2 p_2), \qquad a = (1, 1/2),
$$

where $\psi$ is one of the following scalar CGFs. All have $\psi''(0) = 1$, so all four noises have the same covariance $\mathrm{diag}(1, 1/4)$, and differ only in their higher cumulants:

| noise | jump distribution | $\psi(u)$ | domain of $\psi$ |
|---|---|---|---|
| Gaussian | none | $u^2/2$ | $\mathbb{R}$ |
| type A | jumps of size $\pm 1$ | $\cosh u - 1$ | $\mathbb{R}$ |
| type B | Laplace (exponential tails) | $\dfrac{u^2}{2(1-u^2)}$ | $(-1, 1)$ |
| type C | $\propto \lvert y\rvert^{-1-\alpha} e^{-\lvert y\rvert}$ | $\dfrac{(1+u)^\alpha + (1-u)^\alpha - 2}{2\alpha(\alpha-1)}$ | $[-1, 1]$ |

### Large deviations and optimal control

As $\varepsilon \to 0$, $\mathbb{P}[f(X^\varepsilon_T) \geq z] \asymp \exp (-I(z)/\varepsilon)$. The rate function is the minimal cost of a noise realization $\eta(t)$ that drives the system to the target,

$$
I(z) = \min_\eta \int_0^T L(\eta(t)) dt \quad \text{s.t.} \quad \dot\phi = b(\phi) + \eta,\ \ \phi(0) = X_0,\ \ f(\phi(T)) = z,
$$

where $L = \Psi^{\ast}$ is the Legendre transform of the CGF. Here, $L(\eta) = \sum_i \psi^{\ast}(\eta_i/a_i)$. For Gaussian noise, $L$ is quadratic. For types B and C, $L$ grows only linearly, so large noise is comparatively cheap, and the optimal path may (in principle) contain jumps. The minimizer is the instanton.

### Time reparameterization (TR)

Jumps cannot be represented on a fixed time grid. We therefore introduce an artificial time $s \in [0,1]$, a time density $\kappa(s) = dt/ds \geq 0$, and a rescaled control $\hat\eta = \kappa \eta$, and solve

$$
I(z) = \min_{\hat\eta, \kappa} \int_0^1 \kappa L \left(\hat\eta/\kappa\right) ds
\quad \text{s.t.} \quad
q' = \kappa b(q) + \hat\eta,\ \ q(0) = X_0,\ \ f(q(1)) = z,\ \ \int_0^1 \kappa ds = T.
$$

Physical time is $t = \tau(s) = \int_0^s \kappa ds$. On an interval where $\kappa = 0$ but $q' \neq 0$, the state moves while physical time is frozen: this is a jump of the instanton. Its cost, $\lim_{\kappa\to 0} \kappa L(\hat\eta/\kappa) = \sum_i |\hat\eta_i|/a_i$, is finite for types B and C, and infinite for Gaussian and type A noise. For type C with $\alpha = 1/2$, the instanton jumps for $z \gtrsim 3.77$ (and for $\alpha = 3/2$ for $z \gtrsim 0.86$), whereas type B instantons never jump (Sec. 3.3 of the paper).

### Prefactor

For a continuous instanton, the sharp asymptotics of the tail probability is

$$
\mathbb{P}[f(X^\varepsilon_T) \geq z] \sim \sqrt{\frac{\varepsilon}{2\pi}} \frac{C(z)}{\lambda_z} e^{-I(z)/\varepsilon}, \qquad \lambda_z = I'(z),
$$

and similarly $\rho_\varepsilon(z) \sim (2\pi\varepsilon)^{-1/2} C(z) e^{-I(z)/\varepsilon}$ for the PDF of $f(X^\varepsilon_T)$. The prefactor $C(z)$ follows from a forward Riccati equation along the instanton in physical time, which is the same as for Gaussian noise, except that the noise covariance is replaced by the covariance $\nabla^2\Psi(p)$ of the exponentially tilted noise along the instanton (Sec. 4.1 of the paper).

## Details

1. **Discretization.** The TR problem is discretized on $N = 500$ uniform cells in $s$, with $\hat\eta$ and $\kappa$ constant on each cell, and the state is integrated with Heun's method.
2. **Legendre transform.** For Gaussian and type A noise, $\psi^{\ast}$ is known in closed form. For types B and C, the maximizer $u$ in $\psi^{\ast}(s) = us - \psi(u)$ is found by bisection on $(-1, 1)$, and derivatives use the implicit relation $du/ds = 1/\psi''(u)$ (a `jax.custom_jvp`) instead of differentiating through the loop.
3. **Optimization.** The two equality constraints are enforced with an augmented Lagrangian method. Each subproblem is solved with L-BFGS-B (from SciPy, with the bound $\kappa \geq 0$), and gradients come from JAX automatic differentiation. The final multiplier of the terminal constraint is $\lambda_z = I'(z)$.
4. **Fixing the reparameterization.** The TR problem is invariant under reparameterizations of $s$, so its minimizers are not unique. We add a penalty on deviations of the speed $(\kappa^2 + |q'|^2)^{1/2}$ from its mean, which favors parameterizing the physical graph $\{(\tau(s), q(s))\}$ by arclength.
5. **Continuation.** The noise types are solved in the order Gaussian → A → B → C, each initialized with the previous instanton.
6. **Prefactor.** For continuous instantons, $C(z)$ is computed in physical time along the TR solution. Because the Riccati solution has a pole for types B and C in this example, we integrate the equivalent linearized Hamiltonian ("Jacobi field") equations instead. No prefactor is reported for jumping instantons: the standard Gaussian-fluctuation formula does not apply there (see Sec. 4.3 of the paper).

## Usage

Requirements: Python 3 with `jax`, `numpy`, `scipy` and `matplotlib` (tested with JAX 0.6.2, CPU).

```sh
pip install jax numpy scipy matplotlib
python tr_instantons_2d.py            # z = 4, type C with alpha = 1/2
python tr_instantons_2d.py 3 1.5      # z = 3, type C with alpha = 3/2
```

The script prints the rate function $I(z)$, the multiplier $\lambda_z = I'(z)$, the prefactors $C(z)$ and $C(z)/\lambda_z$, and the size of the jump (if any) for each noise type. It also saves a figure `instantons_z<z>.png` with (a) the instantons in the $(x, y)$-plane, (b) their components over physical time $t$, (c) the same over artificial time $s$, and (d) physical time $\tau(s)$ over artificial time. In (d), flat parts mean frozen physical time, i.e., jumps.

