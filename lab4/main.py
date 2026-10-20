"""Лабораторна робота №4. Точність формул для чисельного диференціювання.
Метод Рунге–Ромберга. Метод Ейткена.

Швидкість зміни вологості ґрунту M(t) = 50 e^(-0.1 t) + 5 sin(t) у точці t0 = 1.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X0 = 1.0
H = 1e-3  # крок для методів Рунге–Ромберга та Ейткена


def out(name):
    return os.path.join(BASE_DIR, name)


def f(t):
    """Вологість ґрунту M(t)."""
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)


def df_exact(t):
    """Точна похідна M'(t) = -5 e^(-0.1 t) + 5 cos(t)."""
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)


def central(x0, h):
    """Центральна різницева формула y'(h) = (f(x0 + h) - f(x0 - h)) / (2h), порядок O(h^2)."""
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def runge_romberg(d_h, d_2h, p=2):
    """Уточнення за Рунге–Ромбергом: y'_R = y'(h) + (y'(h) - y'(2h)) / (2^p - 1)."""
    return d_h + (d_h - d_2h) / (2 ** p - 1)


def aitken(d_h, d_2h, d_4h):
    """Уточнення за Ейткеном і оцінка порядку точності p."""
    value = (d_2h ** 2 - d_4h * d_h) / (2 * d_2h - (d_4h + d_h))
    p = math.log(abs((d_4h - d_2h) / (d_2h - d_h))) / math.log(2)
    return value, p


def main():
    exact = df_exact(X0)
    print(f"1. M'(t) = -5e^(-0.1t) + 5cos(t);  M'({X0:g}) = {exact:.12f}")

    # 2. Залежність похибки від кроку
    hs = np.logspace(-20, 3, 231)
    errors = np.array([abs(central(X0, h) - exact) for h in hs])
    k = int(np.argmin(errors))
    h0, r0 = hs[k], errors[k]
    print(f"\n2. Оптимальний крок h0 = {h0:.0e}, похибка R0 = {r0:.3e}")
    print("   h        | y'(h)              | R = |y'(h) - y'(x0)|")
    for e in range(-20, 4):
        h = 10.0 ** e
        print(f"   {h:8.0e} | {central(X0, h):18.12f} | {abs(central(X0, h) - exact):.3e}")
    with open(out("errors.txt"), "w", encoding="utf-8") as fh:
        fh.write("# h y'(h) R\n")
        for h, r in zip(hs, errors):
            fh.write(f"{h:.6e} {central(X0, h):.15f} {r:.6e}\n")

    # 3–5. Два кроки h і 2h
    d_h, d_2h, d_4h = central(X0, H), central(X0, 2 * H), central(X0, 4 * H)
    r1 = abs(d_h - exact)
    print(f"\n3–5. h = {H:g}: y'(h) = {d_h:.12f}, y'(2h) = {d_2h:.12f}, R1 = {r1:.3e}")

    # 6. Рунге–Ромберг
    d_r = runge_romberg(d_h, d_2h)
    r2 = abs(d_r - exact)
    print(f"6. Рунге–Ромберг: y'_R = {d_r:.12f}, R2 = {r2:.3e} (зменшення в {r1 / r2:.0f} разів)")

    # 7. Ейткен
    d_e, p = aitken(d_h, d_2h, d_4h)
    r3 = abs(d_e - exact)
    print(f"7. y'(4h) = {d_4h:.12f}")
    print(f"   Ейткен: y'_E = {d_e:.12f}, R3 = {r3:.3e}, порядок точності p = {p:.4f}")

    # Те саме для більшого кроку, де похибка методу переважає похибку округлення
    print("\nПорівняння для різних кроків h:")
    print("   h      |   R1 (центр.)  |  R2 (Р–Р)   |  R3 (Ейткен) |   p")
    rows = []
    for h in (1e-1, 5e-2, 1e-2, 1e-3):
        a, b, c = central(X0, h), central(X0, 2 * h), central(X0, 4 * h)
        e_val, e_p = aitken(a, b, c)
        rows.append((h, abs(a - exact), abs(runge_romberg(a, b) - exact), abs(e_val - exact), e_p))
        print(f"   {h:6g} | {rows[-1][1]:.3e} | {rows[-1][2]:.3e} | {rows[-1][3]:.3e} | {e_p:.4f}")

    # Графіки
    plt.figure(figsize=(10, 5))
    plt.loglog(hs, np.maximum(errors, 1e-17))
    plt.loglog(h0, r0, "r*", ms=14, label=f"оптимум: h0 = {h0:.0e}, R0 = {r0:.1e}")
    plt.axvline(H, color="g", ls="--", lw=1, label=f"h = {H:g}")
    plt.xlabel("Крок h")
    plt.ylabel("R = |y'(h) − y'(x0)|")
    plt.title("Похибка центральної різницевої формули залежно від кроку")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("error_vs_h.png"), dpi=120)

    tt = np.linspace(0, 20, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(tt, [f(t) for t in tt])
    ax1.plot(X0, f(X0), "ro")
    ax1.set_title("Вологість ґрунту M(t)")
    ax1.set_xlabel("t")
    ax1.grid(True)
    ax2.plot(tt, [df_exact(t) for t in tt])
    ax2.plot(X0, exact, "ro", label=f"M'(1) = {exact:.6f}")
    ax2.set_title("Швидкість зміни вологості M'(t)")
    ax2.set_xlabel("t")
    ax2.grid(True)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out("function.png"), dpi=120)

    plt.figure(figsize=(10, 4.5))
    hh = [r[0] for r in rows]
    for idx, label in ((1, "центральна різниця"), (2, "Рунге–Ромберг"), (3, "Ейткен")):
        plt.loglog(hh, [r[idx] for r in rows], "o-", label=label)
    plt.xlabel("Крок h")
    plt.ylabel("Абсолютна похибка")
    plt.title("Порівняння точності методів")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("methods.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
