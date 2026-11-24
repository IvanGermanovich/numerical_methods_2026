"""Лабораторна робота №8. Чисельні методи розв'язування нелінійних рівнянь з одним невідомим.

Частина 1: трансцендентне рівняння F(x) = sin(x) - 0.3x - 0.2 = 0 на [0; 4] — методи простої
ітерації, Ньютона, Чебишева, хорд, парабол і зворотної інтерполяції.
Частина 2: алгебраїчне рівняння F(x) = x^3 - 3x^2 - 8x - 10 = 0 (коефіцієнти у файлі poly.txt) —
метод Ньютона за схемою Горнера (дійсний корінь) і метод Ліна (комплексні корені).
"""
import cmath
import math
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
A, B, H = 0.0, 4.0, 0.1
EPS = 1e-10
MAX_ITER = 1000


def out(name):
    return os.path.join(BASE_DIR, name)


# -------------------------------------------------
# Частина 1. Трансцендентне рівняння
# -------------------------------------------------
def F(x):
    return math.sin(x) - 0.3 * x - 0.2


def dF(x):
    return math.cos(x) - 0.3


def d2F(x):
    return -math.sin(x)


def tabulate(a, b, h, filename):
    """Табуляція F(x) на [a; b] з кроком h, запис у файл і пошук змін знаку."""
    n = int(round((b - a) / h))
    xs = [a + i * h for i in range(n + 1)]
    ys = [F(x) for x in xs]
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("# x F(x)\n")
        for x, y in zip(xs, ys):
            fh.write(f"{x:.1f} {y: .10f}\n")
    brackets = [(xs[i], xs[i + 1]) for i in range(n) if ys[i] * ys[i + 1] < 0]
    return xs, ys, brackets


def stop(x_new, x_old, eps=EPS):
    """Критерій: |F(x_{n+1})| < eps і |x_{n+1} - x_n| < eps."""
    return abs(F(x_new.real if isinstance(x_new, complex) else x_new)) < eps and abs(x_new - x_old) < eps


def simple_iteration(x0, eps=EPS):
    """Метод простої релаксації x_{n+1} = x_n + τF(x_n), τ = -1/F'(x0) (умова -2 < τF' < 0)."""
    tau = -1 / dF(x0)
    x, hist = x0, [x0]
    for _ in range(MAX_ITER):
        x_new = x + tau * F(x)
        hist.append(x_new)
        if stop(x_new, x, eps):
            return x_new, hist
        x = x_new
    return x, hist


def newton(x0, eps=EPS):
    """x_{n+1} = x_n - F(x_n)/F'(x_n)."""
    x, hist = x0, [x0]
    for _ in range(MAX_ITER):
        x_new = x - F(x) / dF(x)
        hist.append(x_new)
        if stop(x_new, x, eps):
            return x_new, hist
        x = x_new
    return x, hist


def chebyshev(x0, eps=EPS):
    """x_{n+1} = x_n - F/F' - F^2 F'' / (2 F'^3)."""
    x, hist = x0, [x0]
    for _ in range(MAX_ITER):
        f, f1 = F(x), dF(x)
        x_new = x - f / f1 - f * f * d2F(x) / (2 * f1 ** 3)
        hist.append(x_new)
        if stop(x_new, x, eps):
            return x_new, hist
        x = x_new
    return x, hist


def chords(x0, x1, eps=EPS):
    """Метод хорд (січних): x_{n+1} = x_n - F(x_n)(x_n - x_{n-1}) / (F(x_n) - F(x_{n-1}))."""
    hist = [x0, x1]
    for _ in range(MAX_ITER):
        x_new = x1 - F(x1) * (x1 - x0) / (F(x1) - F(x0))
        hist.append(x_new)
        if stop(x_new, x1, eps):
            return x_new, hist
        x0, x1 = x1, x_new
    return x1, hist


def parabolas(x0, x1, x2, eps=EPS):
    """Метод парабол (Мюллера): корінь квадратичного інтерполяційного многочлена Ньютона."""
    hist = [x0, x1, x2]
    for _ in range(MAX_ITER):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        d1 = (f2 - f1) / (x2 - x1)                  # F(x_n, x_{n-1})
        d2 = (d1 - (f1 - f0) / (x1 - x0)) / (x2 - x0)  # F(x_n, x_{n-1}, x_{n-2})
        w = d1 + (x2 - x1) * d2
        root = cmath.sqrt(w * w - 4 * f2 * d2)
        den = w + root if abs(w + root) >= abs(w - root) else w - root  # менший за модулем Δ
        delta = -2 * f2 / den
        x_new = x2 + (delta.real if abs(delta.imag) < 1e-15 else delta)
        hist.append(x_new)
        if stop(x_new, x2, eps):
            return x_new, hist
        x0, x1, x2 = x1, x2, x_new
    return x2, hist


def inverse_interpolation(x0, x1, x2, eps=EPS):
    """Зворотна інтерполяція: x_{n+1} = L_2(0), многочлен Лагранжа x = L(y) за трьома точками."""
    hist = [x0, x1, x2]
    for _ in range(MAX_ITER):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        x_new = (y1 * y2 / ((y0 - y1) * (y0 - y2)) * x0
                 + y0 * y2 / ((y1 - y0) * (y1 - y2)) * x1
                 + y0 * y1 / ((y2 - y0) * (y2 - y1)) * x2)
        hist.append(x_new)
        if stop(x_new, x2, eps):
            return x_new, hist
        x0, x1, x2 = x1, x2, x_new
    return x2, hist


# -------------------------------------------------
# Частина 2. Алгебраїчне рівняння
# -------------------------------------------------
def read_poly(filename):
    """Зчитує коефіцієнти a_0, a_1, ..., a_m з текстового файлу."""
    with open(filename, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                return [float(v) for v in line.split()]
    raise ValueError("немає коефіцієнтів")


def poly_value(a, x):
    """Значення многочлена за схемою Горнера: b_m = a_m, b_i = a_i + x b_{i+1}, F(x) = b_0."""
    b = a[-1]
    for coef in reversed(a[:-1]):
        b = coef + x * b
    return b


def newton_horner(a, x0, eps=EPS):
    """Метод Ньютона з обчисленням F = b_0 і F' = c_1 за схемою Горнера."""
    x, hist = x0, [x0]
    m = len(a) - 1
    for _ in range(MAX_ITER):
        b = [0.0] * (m + 1)
        c = [0.0] * (m + 1)
        b[m] = c[m] = a[m]
        for i in range(m - 1, -1, -1):
            b[i] = a[i] + x * b[i + 1]
            if i >= 1:
                c[i] = b[i] + x * c[i + 1]
        x_new = x - b[0] / c[1]
        hist.append(x_new)
        if abs(poly_value(a, x_new)) < eps and abs(x_new - x) < eps:
            return x_new, hist
        x = x_new
    return x, hist


def lin(a, alpha0, beta0, eps=EPS):
    """Метод Ліна: F(x) = (x^2 + px + q)(b_2 + ... + b_m x^(m-2)) + b_1 x + b_0, b_1 = b_0 = 0."""
    m = len(a) - 1
    alpha, beta = alpha0, beta0
    hist = [(alpha, beta)]
    for _ in range(MAX_ITER):
        p, q = -2 * alpha, alpha ** 2 + beta ** 2
        b = [0.0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] - p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = a[i] - p * b[i + 1] - q * b[i + 2]
        q1 = a[0] / b[2]
        p1 = (a[1] * b[2] - a[0] * b[3]) / b[2] ** 2
        alpha_new = -p1 / 2
        beta_new = math.sqrt(q1 - alpha_new ** 2)
        hist.append((alpha_new, beta_new))
        if abs(alpha_new - alpha) <= eps and abs(beta_new - beta) <= eps:
            return alpha_new, beta_new, hist
        alpha, beta = alpha_new, beta_new
    return alpha, beta, hist


def main():
    # 1. Табуляція і виділення коренів
    xs, ys, brackets = tabulate(A, B, H, out("tabulation.txt"))
    print(f"1. F(x) = sin(x) - 0.3x - 0.2, табуляція на [{A:g}; {B:g}] з кроком {H} -> tabulation.txt")
    print("   Зміни знаку F(x):")
    for a, b in brackets:
        kind = "F зростає" if F(b) > F(a) else "F спадає"
        print(f"   [{a:.1f}; {b:.1f}]: F({a:.1f}) = {F(a): .4f}, F({b:.1f}) = {F(b): .4f}  ({kind})")

    # 2–4. Уточнення кожного кореня шістьма методами
    results = []
    for a, b in brackets:
        m = (a + b) / 2
        runs = [
            ("Проста ітерація", simple_iteration(m)),
            ("Ньютона", newton(m)),
            ("Чебишева", chebyshev(m)),
            ("Хорд", chords(a, b)),
            ("Парабол", parabolas(a, m, b)),
            ("Зворотної інтерполяції", inverse_interpolation(a, m, b)),
        ]
        print(f"\n2–4. Корінь на [{a:.1f}; {b:.1f}] (eps = {EPS:g}), початкове наближення x0 = {m:.2f}:")
        print("   Метод                   | корінь              | |F(x)|     | ітерацій")
        for name, (root, hist) in runs:
            r = root.real if isinstance(root, complex) else root
            n_start = 1 if name in ("Проста ітерація", "Ньютона", "Чебишева") else (2 if name == "Хорд" else 3)
            print(f"   {name:24s}| {r:.15f} | {abs(F(r)):.2e} | {len(hist) - n_start}")
        results.append(((a, b), runs))

    # 5–7. Алгебраїчне рівняння
    a_poly = read_poly(out("poly.txt"))
    print("\n5–7. Алгебраїчне рівняння, коефіцієнти з poly.txt:", a_poly)
    print("   F(x) = x^3 - 3x^2 - 8x - 10 = (x - 5)(x^2 + 2x + 2): один дійсний і два комплексно-спряжені корені")
    bound = 1 + max(abs(v) for v in a_poly[:-1]) / abs(a_poly[-1])
    print(f"   Усі корені лежать у крузі |x| ≤ 1 + max|a_i|/|a_m| = {bound:g}")
    for x in (0, 1, 5):
        print(f"   F({x}) за схемою Горнера = {poly_value(a_poly, x):g}")

    # 8. Ньютон за схемою Горнера
    x0 = 6.0
    root, hist = newton_horner(a_poly, x0)
    print(f"\n8. Метод Ньютона за схемою Горнера, x0 = {x0:g}:")
    for k, x in enumerate(hist):
        print(f"   x_{k} = {x:.15f}")
    print(f"   Дійсний корінь x = {root:.15f}, F(x) = {poly_value(a_poly, root):.2e}, ітерацій: {len(hist) - 1}")

    # 9. Метод Ліна
    alpha0, beta0 = -0.5, 1.5
    alpha, beta, lhist = lin(a_poly, alpha0, beta0)
    z = complex(alpha, beta)
    print(f"\n9. Метод Ліна, початкове наближення α0 ± iβ0 = {alpha0:g} ± {beta0:g}i:")
    for k in (1, 2, 5, 10, 20, len(lhist) - 1):
        al, be = lhist[k]
        print(f"   k = {k:3d}: α = {al: .12f}, β = {be:.12f}")
    print(f"   Комплексні корені: {alpha:.12f} ± {beta:.12f}i, |F(α + iβ)| = {abs(poly_value(a_poly, z)):.2e}, "
          f"ітерацій: {len(lhist) - 1}")

    # Графіки
    xx = np.linspace(A, B, 400)
    plt.figure(figsize=(10, 4.5))
    plt.plot(xx, [F(x) for x in xx], label="F(x) = sin x − 0,3x − 0,2")
    plt.axhline(0, color="k", lw=0.8)
    plt.plot(xs, ys, "k.", ms=4, label="табуляція, h = 0,1")
    for (a, b), runs in results:
        r = runs[1][1][0]
        plt.plot(r, 0, "r*", ms=14)
        plt.annotate(f"x* = {r:.6f}", (r, 0), textcoords="offset points", xytext=(5, 10))
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Трансцендентна функція та її корені")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("function.png"), dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, ((a, b), runs) in zip(axes, results):
        exact = runs[1][1][0]
        for name, (root, hist) in runs:
            errs = [max(abs((x.real if isinstance(x, complex) else x) - exact), 1e-17) for x in hist]
            ax.semilogy(range(len(errs)), errs, "o-", ms=3, label=name)
        ax.set_title(f"Корінь на [{a:.1f}; {b:.1f}]")
        ax.set_xlabel("Номер наближення n")
        ax.grid(True, which="both", alpha=0.4)
    axes[0].set_ylabel("|x_n − x*|")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("convergence.png"), dpi=120)

    xx = np.linspace(-2, 6, 400)
    plt.figure(figsize=(10, 4.5))
    plt.plot(xx, [poly_value(a_poly, x) for x in xx], label="F(x) = x³ − 3x² − 8x − 10")
    plt.axhline(0, color="k", lw=0.8)
    plt.plot(root, 0, "r*", ms=14, label=f"дійсний корінь {root:.6f}")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title(f"Алгебраїчне рівняння; комплексні корені {alpha:.4f} ± {beta:.4f}i")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("polynomial.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
