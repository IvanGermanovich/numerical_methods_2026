"""Лабораторна робота №7. Ітераційні методи розв'язку систем лінійних алгебраїчних рівнянь.

Методи простої ітерації, Якобі та Зейделя для системи AX = f розмірності 100×100
з діагональним переважанням і точним розв'язком x_i = 2.5; початкове наближення x_i = 1.
"""
import math
import os
import random

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N = 100
X_TRUE = 2.5
X_START = 1.0
EPS0 = 1e-14
MAX_ITER = 10000
SEED = 2026
DOMINANCE = 1.5  # a_ii = 1.5 · Σ_{j≠i} |a_ij|


def out(name):
    return os.path.join(BASE_DIR, name)


# -------------------------------------------------
# 1. Генерування системи з діагональним переважанням
# -------------------------------------------------
def generate_matrix(n, seed=SEED):
    """Позадіагональні елементи — випадкові з [-1; 1], діагональні — 1.5·Σ|a_ij|."""
    rng = random.Random(seed)
    a = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a[i][i] = DOMINANCE * math.fsum(abs(a[i][j]) for j in range(n) if j != i)
    return a


def write_matrix(filename, m):
    with open(filename, "w", encoding="utf-8") as fh:
        for row in m:
            fh.write(" ".join(f"{v:.17e}" for v in row) + "\n")


def read_matrix(filename):
    with open(filename, encoding="utf-8") as fh:
        return [[float(v) for v in line.split()] for line in fh if line.strip()]


def write_vector(filename, v):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"{x:.17e}" for x in v) + "\n")


def read_vector(filename):
    with open(filename, encoding="utf-8") as fh:
        return [float(line) for line in fh if line.strip()]


# -------------------------------------------------
# 2. Допоміжні функції
# -------------------------------------------------
def mat_vec(a, x):
    """Добуток матриці на вектор."""
    return [math.fsum(aij * xj for aij, xj in zip(row, x)) for row in a]


def vec_norm(v):
    """Норма вектора ||v|| = max |v_i|."""
    return max(abs(x) for x in v)


def mat_norm(a):
    """Норма матриці ||A|| = max_i Σ_j |a_ij|."""
    return max(math.fsum(abs(v) for v in row) for row in a)


def residual(a, x, f):
    return vec_norm([ax - fi for ax, fi in zip(mat_vec(a, x), f)])


def residual_floor(a, x):
    """Нев'язка, нижче якої не можна опуститися в подвійній точності: ε_m · max_i Σ|a_ij x_j|."""
    return 2.220446049250313e-16 * max(math.fsum(abs(aij * xj) for aij, xj in zip(row, x)) for row in a)


def converged(a, x_new, x_old, f, eps, floor):
    """Умови закінчення: ||X(k+1) - X(k)|| < eps і ||AX(k+1) - f|| ≤ max(eps, межа округлення)."""
    dx = vec_norm([p - q for p, q in zip(x_new, x_old)])
    return dx < eps and residual(a, x_new, f) <= max(eps, floor), dx


# -------------------------------------------------
# 3. Ітераційні методи
# -------------------------------------------------
def simple_iteration(a, f, x0, tau, eps=EPS0, max_iter=MAX_ITER):
    """X(k+1) = X(k) - τ (A X(k) - f)."""
    x, floor, history = list(x0), residual_floor(a, [X_TRUE] * len(f)), []
    for k in range(1, max_iter + 1):
        ax = mat_vec(a, x)
        x_new = [xi - tau * (axi - fi) for xi, axi, fi in zip(x, ax, f)]
        done, dx = converged(a, x_new, x, f, eps, floor)
        history.append((k, dx, vec_norm([v - X_TRUE for v in x_new])))
        x = x_new
        if done:
            break
    return x, history


def jacobi(a, f, x0, eps=EPS0, max_iter=MAX_ITER):
    """x_i(k+1) = (f_i - Σ_{j≠i} a_ij x_j(k)) / a_ii."""
    n = len(f)
    x, floor, history = list(x0), residual_floor(a, [X_TRUE] * n), []
    for k in range(1, max_iter + 1):
        x_new = [(f[i] - math.fsum(a[i][j] * x[j] for j in range(n) if j != i)) / a[i][i] for i in range(n)]
        done, dx = converged(a, x_new, x, f, eps, floor)
        history.append((k, dx, vec_norm([v - X_TRUE for v in x_new])))
        x = x_new
        if done:
            break
    return x, history


def seidel(a, f, x0, eps=EPS0, max_iter=MAX_ITER):
    """x_i(k+1) = (f_i - Σ_{j<i} a_ij x_j(k+1) - Σ_{j>i} a_ij x_j(k)) / a_ii."""
    n = len(f)
    x, floor, history = list(x0), residual_floor(a, [X_TRUE] * n), []
    for k in range(1, max_iter + 1):
        x_new = list(x)
        for i in range(n):
            x_new[i] = (f[i] - math.fsum(a[i][j] * x_new[j] for j in range(n) if j != i)) / a[i][i]
        done, dx = converged(a, x_new, x, f, eps, floor)
        history.append((k, dx, vec_norm([v - X_TRUE for v in x_new])))
        x = x_new
        if done:
            break
    return x, history


# Норми матриць переходу C для перевірки достатніх умов збіжності
def norm_c_simple(a, tau):
    n = len(a)
    return max(math.fsum(abs((1.0 if i == j else 0.0) - tau * a[i][j]) for j in range(n)) for i in range(n))


def norm_c_jacobi(a):
    n = len(a)
    return max(math.fsum(abs(a[i][j]) for j in range(n) if j != i) / abs(a[i][i]) for i in range(n))


def main():
    # 1. Генерування і запис системи
    a_gen = generate_matrix(N)
    f_gen = mat_vec(a_gen, [X_TRUE] * N)
    write_matrix(out("matrix_A.txt"), a_gen)
    write_vector(out("vector_B.txt"), f_gen)
    print(f"1. Матриця {N}×{N}: a_ij ∈ [-1; 1], a_ii = {DOMINANCE}·Σ|a_ij| (seed = {SEED}), x_i = {X_TRUE}")

    # 2–3. Зчитування, початкове наближення
    a = read_matrix(out("matrix_A.txt"))
    f = read_vector(out("vector_B.txt"))
    x0 = [X_START] * N
    a_norm = mat_norm(a)
    tau = 1 / a_norm  # 0 < τ < 2 / ||A||
    print(f"2. ||A|| = {a_norm:.4f}, max|f_i| = {vec_norm(f):.4f}")
    print(f"   Проста ітерація: τ = 1/||A|| = {tau:.6f} (умова 0 < τ < 2/||A|| = {2 / a_norm:.6f})")
    print(f"   ||C|| простої ітерації = {norm_c_simple(a, tau):.4f}")
    print(f"   ||C|| методу Якобі = max Σ|a_ij|/|a_ii| = {norm_c_jacobi(a):.4f} < 1 (діагональне переважання)")
    print(f"3. Початкове наближення x_i(0) = {X_START}; межа нев'язки в подвійній точності "
          f"{residual_floor(a, [X_TRUE] * N):.2e}")

    # 4. Розв'язання трьома методами
    results = {}
    for name, solve in (("Проста ітерація", lambda: simple_iteration(a, f, x0, tau)),
                        ("Якобі", lambda: jacobi(a, f, x0)),
                        ("Зейдель", lambda: seidel(a, f, x0))):
        x, hist = solve()
        results[name] = (x, hist)
        print(f"\n4. {name}: ітерацій = {len(hist)}, ||X(k+1) − X(k)|| = {hist[-1][1]:.3e}, "
              f"||AX − f|| = {residual(a, x, f):.3e}, max|x_i − 2.5| = {hist[-1][2]:.3e}")
        print("   перші 3 компоненти:", ", ".join(f"{v:.15f}" for v in x[:3]))
        for k, dx, err in hist:
            if k in (1, 2, 5, 10, 20, 50, 100, 150) or k == len(hist):
                print(f"   k = {k:4d}: ||ΔX|| = {dx:.3e}, max|x_i − 2.5| = {err:.3e}")
        write_vector(out(f"solution_{['simple', 'jacobi', 'seidel'][len(results) - 1]}.txt"), x)

    print("\nКількість ітерацій для eps0 = 1e-14:")
    for name, (_, hist) in results.items():
        print(f"   {name:16s}: {len(hist)}")

    plt.figure(figsize=(10, 5))
    for name, (_, hist) in results.items():
        plt.semilogy([0] + [h[0] for h in hist], [abs(X_START - X_TRUE)] + [max(h[2], 1e-17) for h in hist],
                     label=f"{name} ({len(hist)} ітер.)")
    plt.axhline(EPS0, color="r", ls="--", lw=1, label="eps0 = 1e-14")
    plt.xlabel("Номер ітерації k")
    plt.ylabel("max|x_i − 2,5|")
    plt.title("Збіжність ітераційних методів")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("convergence.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
