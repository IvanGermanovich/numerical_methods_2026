"""Лабораторна робота №5. Складова квадратурна формула для чисельного інтегрування.
Методи підвищення точності. Адаптивний алгоритм.

Накопичене навантаження на сервер: I = ∫ f(x) dx на [0; 24],
f(x) = 50 + 20 sin(πx/12) + 5 e^(-0.2 (x - 12)^2).
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
A, B = 0.0, 24.0
EPS = 1e-12  # задана точність для пошуку N_opt


def out(name):
    return os.path.join(BASE_DIR, name)


def f(x):
    """Інтенсивність навантаження на сервер."""
    return 50 + 20 * math.sin(math.pi * x / 12) + 5 * math.exp(-0.2 * (x - 12) ** 2)


def exact_integral():
    """I0: ∫50 = 1200, ∫20 sin(πx/12) по повному періоду = 0, ∫5e^(-0.2(x-12)^2) = 5√(5π)·erf(12/√5)."""
    return 50 * (B - A) + 5 * math.sqrt(math.pi / 0.2) * math.erf(12 * math.sqrt(0.2))


def simpson(n):
    """Складова формула Сімпсона з n (парним) відрізками розбиття."""
    h = (B - A) / n
    odd = sum(f(A + i * h) for i in range(1, n, 2))
    even = sum(f(A + i * h) for i in range(2, n, 2))
    return h / 3 * (f(A) + 4 * odd + 2 * even + f(B))


def runge_romberg(i_n, i_half, p=4):
    """I_R = I(N0) + (I(N0) - I(N0/2)) / (2^p - 1), для Сімпсона 2^4 - 1 = 15."""
    return i_n + (i_n - i_half) / (2 ** p - 1)


def aitken(i1, i2, i3):
    """Уточнення за Ейткеном за I(N0), I(N0/2), I(N0/4) і оцінка порядку p."""
    value = (i2 ** 2 - i1 * i3) / (2 * i2 - (i1 + i3))
    p = math.log(abs((i3 - i2) / (i2 - i1))) / math.log(2)
    return value, p


def adaptive_simpson(delta):
    """Адаптивний алгоритм на основі формули Сімпсона. Повертає інтеграл і кількість обчислень f."""
    calls = 0

    def fc(x):
        nonlocal calls
        calls += 1
        return f(x)

    def step(x0, x1, y0, ym, y1, whole, tol):
        h = x1 - x0
        xq1, xq3 = x0 + h / 4, x0 + 3 * h / 4
        yq1, yq3 = fc(xq1), fc(xq3)
        left = h / 12 * (y0 + 4 * yq1 + ym)
        right = h / 12 * (ym + 4 * yq3 + y1)
        if abs(whole - (left + right)) <= tol:  # умова збіжності (3)
            return left + right
        xm = (x0 + x1) / 2
        return (step(x0, xm, y0, yq1, ym, left, tol / 2)
                + step(xm, x1, ym, yq3, y1, right, tol / 2))

    y0, ym, y1 = fc(A), fc((A + B) / 2), fc(B)
    whole = (B - A) / 6 * (y0 + 4 * ym + y1)
    return step(A, B, y0, ym, y1, whole, delta), calls


def main():
    i0 = exact_integral()
    print(f"2. Точне значення I0 = {i0:.12f}")

    # 3–4. Залежність похибки від N
    ns = np.arange(10, 1001, 2)
    errs = np.array([abs(simpson(n) - i0) for n in ns])
    n_opt = int(next(n for n, e in zip(ns, errs) if e <= EPS))
    eps_opt = abs(simpson(n_opt) - i0)
    print(f"\n4. N_opt = {n_opt} (перше N з ε(N) ≤ {EPS:g}), epsopt = {eps_opt:.3e}")
    for n in (10, 20, 50, 100, 200, 500, 1000):
        print(f"   N = {n:4d}: I(N) = {simpson(n):.12f}, ε = {abs(simpson(n) - i0):.3e}")
    with open(out("errors.txt"), "w", encoding="utf-8") as fh:
        fh.write("# N I(N) eps(N)\n")
        for n, e in zip(ns, errs):
            fh.write(f"{n} {simpson(int(n)):.15f} {e:.6e}\n")

    # 5. N0 ≈ N_opt / 10, кратне 8
    n0 = max(8, int(round(n_opt / 10 / 8)) * 8)
    i_n0 = simpson(n0)
    eps0 = abs(i_n0 - i0)
    print(f"\n5. N0 = {n0}: I(N0) = {i_n0:.12f}, eps0 = {eps0:.3e}")

    # 6. Рунге–Ромберг
    i_half, i_quarter = simpson(n0 // 2), simpson(n0 // 4)
    i_r = runge_romberg(i_n0, i_half)
    eps_r = abs(i_r - i0)
    print(f"6. I(N0/2) = {i_half:.12f}; Рунге–Ромберг: I_R = {i_r:.12f}, epsR = {eps_r:.3e}")

    # 7. Ейткен
    i_e, p = aitken(i_n0, i_half, i_quarter)
    eps_e = abs(i_e - i0)
    print(f"7. I(N0/4) = {i_quarter:.12f}; Ейткен: I_E = {i_e:.12f}, epsE = {eps_e:.3e}, p = {p:.4f}")

    # 8. Характер зміни похибок для різних N0
    print("\n8. Похибки методів для різних N0:")
    print("   N0 |   eps0     |   epsR     |   epsE     |   p")
    methods = []
    for n in (8, 16, 24, 32, 40):
        a1, a2, a3 = simpson(n), simpson(n // 2), simpson(n // 4)
        ev, ep = aitken(a1, a2, a3)
        methods.append((n, abs(a1 - i0), abs(runge_romberg(a1, a2) - i0), abs(ev - i0), ep))
        print(f"   {n:2d} | {methods[-1][1]:.3e} | {methods[-1][2]:.3e} | {methods[-1][3]:.3e} | {ep:.4f}")

    # 9. Адаптивний алгоритм
    print("\n9. Адаптивний алгоритм:")
    print("   δ       | I                 | ε          | обчислень f")
    adaptive = []
    for k in range(1, 13):
        delta = 10.0 ** -k
        val, calls = adaptive_simpson(delta)
        adaptive.append((delta, val, abs(val - i0), calls))
        print(f"   {delta:7.0e} | {val:.12f} | {abs(val - i0):.3e} | {calls}")
    with open(out("adaptive.txt"), "w", encoding="utf-8") as fh:
        fh.write("# delta I eps calls\n")
        for row in adaptive:
            fh.write(f"{row[0]:.0e} {row[1]:.15f} {row[2]:.6e} {row[3]}\n")

    # Графіки
    xx = np.linspace(A, B, 500)
    plt.figure(figsize=(10, 4.5))
    plt.plot(xx, [f(x) for x in xx])
    plt.fill_between(xx, [f(x) for x in xx], alpha=0.15)
    plt.xlabel("Час x, год")
    plt.ylabel("Навантаження f(x)")
    plt.title(f"Інтенсивність навантаження на сервер; I0 = {i0:.6f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out("function.png"), dpi=120)

    plt.figure(figsize=(10, 4.5))
    plt.loglog(ns, np.maximum(errs, 1e-17))
    plt.loglog(n_opt, eps_opt, "r*", ms=14, label=f"N_opt = {n_opt}, ε = {eps_opt:.1e}")
    plt.axhline(EPS, color="r", ls="--", lw=1, label="ε = 1e-12")
    plt.xlabel("Число розбиття N")
    plt.ylabel("ε(N) = |I(N) − I0|")
    plt.title("Похибка складової формули Сімпсона")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("error_vs_n.png"), dpi=120)

    plt.figure(figsize=(10, 4.5))
    n0s = [r[0] for r in methods]
    for idx, label in ((1, "Сімпсон, I(N0)"), (2, "Рунге–Ромберг"), (3, "Ейткен")):
        plt.semilogy(n0s, np.maximum([r[idx] for r in methods], 1e-17), "o-", label=label)
    plt.xticks(n0s)
    plt.xlabel("N0")
    plt.ylabel("Абсолютна похибка")
    plt.title("Порівняння точності методів")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("methods.png"), dpi=120)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ds = [r[0] for r in adaptive]
    ax1.loglog(ds, np.maximum([r[2] for r in adaptive], 1e-17), "o-", label="фактична похибка ε")
    ax1.loglog(ds, ds, "k--", lw=1, label="ε = δ")
    ax1.invert_xaxis()
    ax1.set_xlabel("δ")
    ax1.set_ylabel("ε")
    ax1.set_title("Точність адаптивного алгоритму")
    ax1.grid(True, which="both", alpha=0.4)
    ax1.legend()
    ax2.semilogx(ds, [r[3] for r in adaptive], "s-")
    ax2.invert_xaxis()
    ax2.set_xlabel("δ")
    ax2.set_ylabel("Кількість обчислень f(x)")
    ax2.set_title("Обчислювальні витрати")
    ax2.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out("adaptive.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
