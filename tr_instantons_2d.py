import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------
Z = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0        # target value of the observable
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5    # exponent for type C noise

T = 1.0                          # final time
X0 = jnp.zeros(2)                # initial condition
G = jnp.array([1.0, 2.0])        # observable f(x, y) = x + 2 y = <G, (x, y)>
AMP = jnp.array([1.0, 0.5])      # noise amplitudes a_i

N = 500                          # number of grid cells in artificial time
DS = 1.0 / N                     # artificial time step
KAPPA_MIN = 1e-5                 # kappa L(hat_eta / kappa) is evaluated at max(kappa, KAPPA_MIN)
BETA = 0.1                       # weight of the penalty fixing the parameterization of s
MU_VALUES = (10.0, 30.0, 100.0, 300.0, 1000.0)   # augmented Lagrangian penalty parameters


def drift(x):
    return jnp.array([-x[0] - x[0] * x[1], -4.0 * x[1] + x[0] ** 2])


# ----------------------------------------------------------------------------------------------
# Noise types: scalar CGF psi(u), its derivatives, and the Legendre transform psi^*
# ----------------------------------------------------------------------------------------------
def make_noise(kind):
    """Return (psi, d2psi, u_of_s) for kind in {'Gaussian', 'A', 'B', 'C'}, and the maximizer u(s)
    in the Legendre transform psi^*(s) = u(s) s - psi(u(s)), i.e., the solution of psi'(u) = s.

    For Gaussian and type A noise, u(s) is known in closed form. For types B and C, the domain
    of psi is bounded, |u| <= 1, and we find u(s) by bisection. For type C with alpha > 1 and
    |s| >= psi'(1), there is no solution inside the domain, and bisection correctly returns the
    boundary value u = sign(s) ("saturation", where psi^* is affine). Derivatives of u(s) are
    supplied by the implicit relation du/ds = 1 / psi''(u) instead of
    differentiating through the bisection loop.
    """
    if kind == "Gaussian":
        psi, dpsi, d2psi = (lambda u: 0.5 * u ** 2), (lambda u: u), (lambda u: 1.0 + 0.0 * u)
    elif kind == "A":
        psi, dpsi, d2psi = (lambda u: jnp.cosh(u) - 1.0), jnp.sinh, jnp.cosh
    elif kind == "B":
        psi = lambda u: u ** 2 / (2.0 * (1.0 - u ** 2))
        dpsi = lambda u: u / (1.0 - u ** 2) ** 2
        d2psi = lambda u: (1.0 + 3.0 * u ** 2) / (1.0 - u ** 2) ** 3
    elif kind == "C":
        al = ALPHA
        psi = lambda u: ((1 + u) ** al + (1 - u) ** al - 2.0) / (2.0 * al * (al - 1.0))
        dpsi = lambda u: ((1 + u) ** (al - 1) - (1 - u) ** (al - 1)) / (2.0 * (al - 1.0))
        d2psi = lambda u: 0.5 * ((1 + u) ** (al - 2) + (1 - u) ** (al - 2))

    def solve(s):
        if kind == "Gaussian":
            return s
        if kind == "A":
            return jnp.arcsinh(s)

        def bisect(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            below = dpsi(mid) < s
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        one = (1.0 - 1e-15) * jnp.ones_like(s)
        lo, hi = jax.lax.fori_loop(0, 64, bisect, (-one, one))
        return 0.5 * (lo + hi)

    @jax.custom_jvp
    def u_of_s(s):
        return solve(s)

    @u_of_s.defjvp
    def u_of_s_jvp(primals, tangents):
        u = solve(primals[0])
        return u, tangents[0] / d2psi(u)

    return psi, d2psi, u_of_s


def lagrangian(kind):
    """L(eta) = sum_i psi^*(eta_i / a_i), the cost of the noise realization eta."""
    psi, _, u_of_s = make_noise(kind)

    def L(eta):
        s = eta / AMP
        u = u_of_s(s)
        return jnp.sum(u * s - psi(u), axis=-1)

    return L


# ----------------------------------------------------------------------------------------------
# TR control problem
# ----------------------------------------------------------------------------------------------
def integrate(hat_eta, kappa):
    """Solve q' = kappa b(q) + hat_eta on the artificial time grid with Heun's method, for
    controls hat_eta, kappa that are constant on each grid cell. Returns q at the N+1 grid points."""
    def step(q, controls):
        h, k = controls
        q_pred = q + DS * (k * drift(q) + h)
        q_next = q + 0.5 * DS * (k * drift(q) + k * drift(q_pred) + 2.0 * h)
        return q_next, q_next

    _, q = jax.lax.scan(step, X0, (hat_eta, kappa))
    return jnp.concatenate([X0[None], q])


def make_objective(kind):
    L = lagrangian(kind)

    def action(hat_eta, kappa):
        """Discretized int_0^1 kappa L(hat_eta / kappa) ds (perspective). For kappa -> 0, the
        integrand tends to the recession cost L_inf(hat_eta) = sum_i |hat_eta_i| / a_i (types B
        and C), or to infinity (Gaussian and type A). To avoid dividing by zero, we evaluate it
        at max(kappa, KAPPA_MIN), which is very close to L_inf for types B and C."""
        k = jnp.maximum(kappa, KAPPA_MIN)
        return DS * jnp.sum(k * L(hat_eta / k[:, None]))

    def objective(x, mult_f, mult_t, mu):
        """Augmented Lagrangian for the two constraints f(q(1)) = z and int kappa ds = T.

        The TR problem is invariant under reparameterizations of s, so its minimizers are not
        unique. We fix this freedom with a penalty on deviations of the speed in the (t, q) plane,
        v = (kappa^2 + |q'|^2)^(1/2), from its mean, which favors a parameterization by arclength
        of the physical graph {(tau(s), q(s))}."""
        hat_eta, kappa = x[:2 * N].reshape(N, 2), x[2 * N:]
        q = integrate(hat_eta, kappa)
        c_f = G @ q[-1] - Z
        c_t = DS * jnp.sum(kappa) - T
        v = jnp.sqrt(kappa ** 2 + jnp.sum(jnp.diff(q, axis=0) ** 2, axis=1) / DS ** 2 + 1e-16)
        return (action(hat_eta, kappa)
                - mult_f * c_f + 0.5 * mu * c_f ** 2
                - mult_t * c_t + 0.5 * mu * c_t ** 2
                + 0.5 * BETA * DS * jnp.sum((v - jnp.mean(v)) ** 2))

    return jax.jit(jax.value_and_grad(objective)), jax.jit(action)


def solve_tr(kind, x):
    """Solve the TR problem from the initial guess x = (hat_eta, kappa) by an augmented
    Lagrangian method. Each inner problem is solved by L-BFGS-B, with the bound kappa >= 0 and
    gradients from JAX. Returns the instanton and the Lagrange multiplier lambda_z = I'(z)."""
    value_and_grad, action = make_objective(kind)
    bounds = [(None, None)] * (2 * N) + [(0.0, None)] * N
    mult_f, mult_t = 1.0, 0.0
    for mu in MU_VALUES:
        def fun(x):
            value, grad = value_and_grad(x, mult_f, mult_t, mu)
            return float(value), np.asarray(grad)

        x = minimize(fun, x, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=5000, ftol=1e-12, gtol=1e-9)).x
        hat_eta, kappa = x[:2 * N].reshape(N, 2), x[2 * N:]
        q = np.asarray(integrate(hat_eta, kappa))
        mult_f -= mu * (G @ q[-1] - Z)              # multiplier updates
        mult_t -= mu * (DS * kappa.sum() - T)
    return dict(x=x, hat_eta=hat_eta, kappa=kappa, q=q, lam=float(mult_f),
                action=float(action(hat_eta, kappa)),
                s=np.linspace(0.0, 1.0, N + 1),
                t=np.concatenate([[0.0], DS * np.cumsum(kappa)]))


# ----------------------------------------------------------------------------------------------
# Prefactor from the linearized Hamiltonian flow (Riccati equation) in physical time
# ----------------------------------------------------------------------------------------------
def prefactor(kind, sol):
    """PDF prefactor C(z) for a continuous instanton,

        C(z) = exp(1/2 int_0^T tr[H_xx Q] dt) / sqrt(<g, Q(T) g>),
        Q' = Psi''(p) + A Q + Q A^T + Q H_xx Q,   Q(0) = 0,             

    with A = grad b(q), H_xx = Hessian of <p, b(q)> with respect to q, and the covariance
    Psi''(p) of the exponentially tilted noise. The momentum along the instanton is
    p_i = u(eta_i / a_i) / a_i, with the physical noise eta = hat_eta / kappa.

    For types B and C, Q has a pole along the instanton here, so we do not integrate the Riccati
    equation directly, but write Q = X P^{-1} with the linear "Jacobi field" equations

        X' = A X + Psi''(p) P,   P' = -H_xx X - A^T P,   X(0) = 0,   P(0) = Id,

    i.e., the linearization of Hamilton's equations. Since (log det P)' = -tr[H_xx Q] - tr A,
    the prefactor becomes C(z) = exp(-1/2 int_0^T tr A dt) / sqrt(<g, X(T) adj(P(T)) g>), with
    adj(P) = det(P) P^{-1}, which remains finite at poles of Q. We integrate exactly on each TR
    grid cell (matrix exponential with coefficients frozen at the cell midpoint), with physical
    time step dt_k = kappa_k ds.
    """
    _, d2psi, u_of_s = make_noise(kind)
    kappa, q, a, g = sol["kappa"], sol["q"], np.asarray(AMP), np.asarray(G)
    eta = sol["hat_eta"] / np.maximum(kappa, KAPPA_MIN)[:, None]    # physical noise
    u = np.asarray(u_of_s(jnp.asarray(eta / a)))
    p = u / a                                                       # momentum
    psi_pp = a ** 2 * np.asarray(d2psi(jnp.asarray(u)))             # diagonal of Psi''(p)

    XP = np.vstack([np.zeros((2, 2)), np.eye(2)])                   # (X, P) stacked
    trace_A = 0.0
    for k in range(N):
        x, y = 0.5 * (q[k] + q[k + 1])
        A = np.array([[-1.0 - y, -x], [2.0 * x, -4.0]])
        Hxx = np.array([[2.0 * p[k, 1], -p[k, 0]], [-p[k, 0], 0.0]])
        dt = kappa[k] * DS
        XP = expm(dt * np.block([[A, np.diag(psi_pp[k])], [-Hxx, -A.T]])) @ XP
        trace_A += dt * np.trace(A)
    X, P = XP[:2], XP[2:]
    adj_P = np.array([[P[1, 1], -P[0, 1]], [-P[1, 0], P[0, 0]]])
    return np.exp(-0.5 * trace_A) / np.sqrt(g @ X @ adj_P @ g)


# ----------------------------------------------------------------------------------------------
# Main: solve for all four noise types, print results, plot
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    kinds = ["Gaussian", "A", "B", "C"]
    labels = {"Gaussian": "Gaussian", "A": "type A", "B": "type B",
              "C": rf"type C ($\alpha = {ALPHA:g}$)"}

    # initial guess: small noise, uniform physical time. Each subsequent noise type is
    # initialized with the instanton of the previous one (Gaussian -> A -> B -> C), which helps
    # in particular for type C, whose cost is the most poorly conditioned.
    rng = np.random.default_rng(0)
    x = np.concatenate([0.2 + 0.05 * rng.standard_normal(2 * N), np.full(N, T)])

    results = {}
    print(f"z = {Z:g}, type C exponent alpha = {ALPHA:g}\n")
    print(f"{'noise':>10} {'I(z)':>10} {'lambda_z':>10} {'C(z)':>10} {'C/lambda':>10} {'jump':>8}")
    for kind in kinds:
        sol = solve_tr(kind, x)
        x = sol["x"]
        # jump of the instanton = state displacement while physical time is frozen (kappa ~ 0)
        dq = np.linalg.norm(np.diff(sol["q"], axis=0), axis=1)
        sol["jump"] = dq[sol["kappa"] < KAPPA_MIN].sum()
        C = prefactor(kind, sol) if sol["jump"] < 1e-3 else np.nan
        results[kind] = sol
        print(f"{kind:>10} {sol['action']:10.5f} {sol['lam']:10.5f} {C:10.5f} "
              f"{C / sol['lam']:10.5f} {sol['jump']:8.4f}", flush=True)
    print("\nTail probability: P[f(X_T) >= z] ~ sqrt(eps / (2 pi)) * C / lambda * exp(-I / eps).")
    print("For jumping instantons, the prefactor is not computed (nan), see Sec. 4.3 of the paper.")

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    # (a) instantons in the (x, y) plane, with the drift field and the target set
    x_max = max(r["q"][:, 0].max() for r in results.values()) + 0.5
    y_max = max(r["q"][:, 1].max() for r in results.values()) + 0.5
    xs, ys = np.meshgrid(np.linspace(-0.5, x_max, 21), np.linspace(-0.5, y_max, 21))
    ax[0, 0].quiver(xs, ys, -xs - xs * ys, -4 * ys + xs ** 2, color="0.8", angles="xy")
    ax[0, 0].plot([-0.5, x_max], [(Z + 0.5) / 2, (Z - x_max) / 2], "k--", lw=1,
                  label=f"$x + 2y = {Z:g}$")
    for kind in kinds:
        ax[0, 0].plot(*results[kind]["q"].T, label=labels[kind])
    ax[0, 0].set(xlabel="$x$", ylabel="$y$", xlim=(-0.5, x_max), ylim=(-0.5, y_max),
                 title="instantons")
    ax[0, 0].legend()
    # (b, c) components over physical time t and artificial time s
    for i, kind in enumerate(kinds):
        r = results[kind]
        for a, time in ((ax[0, 1], r["t"]), (ax[1, 0], r["s"])):
            a.plot(time, r["q"][:, 0], color=f"C{i}", label=labels[kind])
            a.plot(time, r["q"][:, 1], color=f"C{i}", ls="--")
    ax[0, 1].set(xlabel="physical time $t$", ylabel="$x$ (solid), $y$ (dashed)")
    ax[1, 0].set(xlabel="artificial time $s$", ylabel="$x$ (solid), $y$ (dashed)")
    # (d) physical time t = tau(s) over artificial time; flat parts = frozen time (jumps)
    for i, kind in enumerate(kinds):
        ax[1, 1].plot(results[kind]["s"], results[kind]["t"], color=f"C{i}", label=labels[kind])
    ax[1, 1].set(xlabel="artificial time $s$", ylabel=r"physical time $t = \tau(s)$")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(f"instantons_z{Z:g}.png", dpi=150)
    plt.show()
