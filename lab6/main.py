"""Лабораторна робота №6. LU-розклад. Ітераційні методи уточнення розв'язку СЛАР.

Система AX = B розмірності 100×100 з випадковою матрицею A і точним розв'язком x_i = 2.5.
"""
import math
import os
import random

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N = 100
X_TRUE = 2.5
EPS0 = 1e-14
MAX_ITER = 50
SEED = 2026


def out(name):
    return os.path.join(BASE_DIR, name)


# -------------------------------------------------
# 1. Генерування системи та робота з файлами
# -------------------------------------------------
def generate_matrix(n, seed=SEED):
    """Випадкова матриця n×n з елементами з [-10; 10]."""
    rng = random.Random(seed)
    return [[rng.uniform(-10, 10) for _ in range(n)] for _ in range(n)]


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
# 2. LU-розклад і розв'язання системи
# -------------------------------------------------
def lu_decompose(a):
    """A = LU, L — нижня трикутна, U — верхня трикутна з одиницями на діагоналі."""
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        U[i][i] = 1.0
    for k in range(n):
        # k-й стовпчик L: l_ik = a_ik - Σ_{j<k} l_ij u_jk, i ≥ k
        for i in range(k, n):
            L[i][k] = a[i][k] - sum(L[i][j] * U[j][k] for j in range(k))
        # k-й рядок U: u_ki = (a_ki - Σ_{j<k} l_kj u_ji) / l_kk, i > k
        for i in range(k + 1, n):
            U[k][i] = (a[k][i] - sum(L[k][j] * U[j][i] for j in range(k))) / L[k][k]
    return L, U


def write_lu(filename, L, U):
    """Записує L і U в один текстовий файл."""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("# L\n")
        for row in L:
            fh.write(" ".join(f"{v:.10e}" for v in row) + "\n")
        fh.write("# U\n")
        for row in U:
            fh.write(" ".join(f"{v:.10e}" for v in row) + "\n")


def lu_solve(L, U, b):
    """Розв'язує LZ = B (прямий хід), потім UX = Z (зворотний хід)."""
    n = len(b)
    z = [0.0] * n
    for k in range(n):
        z[k] = (b[k] - sum(L[k][j] * z[j] for j in range(k))) / L[k][k]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(U[k][j] * x[j] for j in range(k + 1, n))
    return x


def mat_vec(a, x):
    """Добуток матриці на вектор (math.fsum — точне підсумовування)."""
    return [math.fsum(aij * xj for aij, xj in zip(row, x)) for row in a]


def norm(v):
    """Норма вектора ||v|| = max |v_i|."""
    return max(abs(x) for x in v)


def residual_norm(a, x, b):
    """eps = max_i |Σ a_ij x_j - b_i|."""
    return norm([ax - bi for ax, bi in zip(mat_vec(a, x), b)])


# -------------------------------------------------
# 5. Ітераційне уточнення
# -------------------------------------------------
def residual_floor(a, x):
    """Нев'язка, нижче якої не можна опуститися в подвійній точності: ε_m · max_i Σ|a_ij x_j|."""
    return 2.220446049250313e-16 * max(math.fsum(abs(aij * xj) for aij, xj in zip(row, x)) for row in a)


def refine(a, L, U, b, x0, eps=EPS0, max_iter=MAX_ITER):
    """X = X0 + ΔX, де AΔX = R = B - AX0; LU-розклад використовується повторно.

    Умова закінчення: ||ΔX|| ≤ eps і ||AX - B|| ≤ max(eps, межа округлення).
    """
    x = list(x0)
    history = []
    floor = residual_floor(a, x0)
    for it in range(1, max_iter + 1):
        r = [bi - ax for bi, ax in zip(b, mat_vec(a, x))]
        dx = lu_solve(L, U, r)
        x = [xi + di for xi, di in zip(x, dx)]
        dx_norm, res = norm(dx), residual_norm(a, x, b)
        history.append((it, dx_norm, res, norm([xi - X_TRUE for xi in x])))
        if dx_norm <= eps and res <= max(eps, floor):
            break
    return x, history


def main():
    # 1. Генерування A, обчислення B, запис у файли
    a_gen = generate_matrix(N)
    b_gen = mat_vec(a_gen, [X_TRUE] * N)
    write_matrix(out("matrix_A.txt"), a_gen)
    write_vector(out("vector_B.txt"), b_gen)
    print(f"1. Згенеровано матрицю {N}×{N} (елементи з [-10; 10], seed = {SEED}), x_i = {X_TRUE}")

    # 2–3. Зчитування, LU-розклад, розв'язання
    a = read_matrix(out("matrix_A.txt"))
    b = read_vector(out("vector_B.txt"))
    L, U = lu_decompose(a)
    write_lu(out("lu.txt"), L, U)
    lu_err = max(abs(math.fsum(L[i][j] * U[j][k] for j in range(N)) - a[i][k])
                 for i in range(N) for k in range(N))
    print(f"2. LU-розклад записано у файл lu.txt; max|LU - A| = {lu_err:.3e}")
    print(f"   min|l_kk| = {min(abs(L[k][k]) for k in range(N)):.4f}")
    x0 = lu_solve(L, U, b)
    write_vector(out("solution_X0.txt"), x0)

    # 4. Точність
    eps = residual_norm(a, x0, b)
    err0 = norm([xi - X_TRUE for xi in x0])
    print(f"3–4. Розв'язок X0: eps = max|Σa_ij x_j - b_i| = {eps:.3e}, max|x_i - 2.5| = {err0:.3e}")
    print("   перші 5 компонент:", ", ".join(f"{v:.15f}" for v in x0[:5]))

    # 5. Уточнення
    x, history = refine(a, L, U, b, x0)
    write_vector(out("solution_X.txt"), x)
    print(f"\n5. Ітераційне уточнення (eps0 = {EPS0:g}):")
    print(" ітерація | ||ΔX||     | ||AX - B|| | max|x_i - 2.5|")
    print(f"     0    |     —      | {eps:.3e}  | {err0:.3e}")
    for it, dxn, res, err in history:
        print(f"    {it:3d}   | {dxn:.3e}  | {res:.3e}  | {err:.3e}")
    print(f"Кількість ітерацій: {len(history)}")
    print(f"max|b_i| = {norm(b):.2f}; межа нев'язки в подвійній точності ε_m·max Σ|a_ij x_j| = "
          f"{residual_floor(a, x):.2e} > eps0, тому для нев'язки eps0 = 1e-14 недосяжна")

    its = [0] + [h[0] for h in history]
    plt.figure(figsize=(10, 4.5))
    plt.semilogy(its, [max(v, 1e-17) for v in [eps] + [h[2] for h in history]], "o-", label="||AX − B||")
    plt.semilogy(its[1:], [max(h[1], 1e-17) for h in history], "s-", label="||ΔX||")
    plt.semilogy(its, [max(v, 1e-17) for v in [err0] + [h[3] for h in history]], "^-", label="max|x_i − 2,5|")
    plt.axhline(EPS0, color="r", ls="--", lw=1, label="eps0 = 1e-14")
    plt.xticks(its)
    plt.xlabel("Ітерація")
    plt.ylabel("Норма")
    plt.title("Ітераційне уточнення розв'язку СЛАР")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("refinement.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
