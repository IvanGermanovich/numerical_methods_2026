"""Лабораторна робота №9. Метод Хука–Дживса багатовимірної оптимізації.

1) Тест: мінімум функції Розенброка f(X) = 100(x1^2 - x2)^2 + (x1 - 1)^2, X(0) = (-1.2; 0).
2) Система нелінійних рівнянь (m = 2):
       f1(x, y) = x^2 + y^2 - 4 = 0       (коло радіуса 2),
       f2(x, y) = y - x^2 + 1 = 0          (парабола y = x^2 - 1),
   розв'язується як мінімум цільової функції Φ(X) = f1^2 + f2^2.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Q = 2.0        # коефіцієнт зменшення кроку
P = 2.0        # коефіцієнт пошуку за зразком
EPS1 = 1e-8    # критерій за кроком
EPS2 = 1e-16   # критерій за зміною цільової функції
STEP0 = 0.5    # початкова величина кроку для всіх змінних
MAX_ITER = 100000


def out(name):
    return os.path.join(BASE_DIR, name)


# -------------------------------------------------
# Цільові функції
# -------------------------------------------------
def rosenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


def f1(x):
    return x[0] ** 2 + x[1] ** 2 - 4


def f2(x):
    return x[1] - x[0] ** 2 + 1


def phi(x):
    """Цільова функція Φ(X) = f1^2 + f2^2 (дорівнює нулю в розв'язку системи)."""
    return f1(x) ** 2 + f2(x) ** 2


# -------------------------------------------------
# Метод Хука–Дживса
# -------------------------------------------------
class Counter:
    """Рахує кількість обчислень цільової функції."""

    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.func(x)


def explore(func, base, f_base, steps, reduce):
    """Досліджуючий пошук навколо точки base.

    Для кожної змінної пробуємо x_i + Δx_i, потім x_i - Δx_i. Якщо жодна не зменшує Φ і reduce = True,
    зменшуємо крок Δx_i = Δx_i / q і повторюємо, доки Δx_i ≥ eps1.
    """
    x = list(base)
    fx = f_base
    for i in range(len(x)):
        while True:
            improved = False
            for sign in (1, -1):
                trial = list(x)
                trial[i] = x[i] + sign * steps[i]
                ft = func(trial)
                if ft < fx:
                    x, fx, improved = trial, ft, True
                    break
            if improved or not reduce:
                break
            steps[i] /= Q
            if steps[i] < EPS1:
                break
    return x, fx


def hooke_jeeves(func, x0, step0=STEP0):
    """Повертає точку мінімуму, значення функції, траєкторію спуску та кількість обчислень."""
    f = Counter(func)
    steps = [step0] * len(x0)
    x_base = list(x0)
    f_base = f(x_base)
    trajectory = [(list(x_base), f_base)]
    for _ in range(MAX_ITER):
        # Досліджуючий пошук
        x1, f1_val = explore(f, x_base, f_base, steps, reduce=True)
        if x1 == x_base:  # крок по всіх змінних став менше eps1
            break
        trajectory.append((list(x1), f1_val))
        if max(steps) < EPS1 and abs(f1_val - f_base) < EPS2:
            x_base, f_base = x1, f1_val
            break
        # Пошук за зразком
        x0_, f0_ = x_base, f_base
        while True:
            xp = [b + P * (b - a) for a, b in zip(x0_, x1)]
            x2, f2_val = explore(f, xp, f(xp), steps, reduce=False)
            if f2_val < f1_val:
                x0_, x1, f1_val = x1, x2, f2_val
                trajectory.append((list(x1), f1_val))
            else:
                break
        x_base, f_base = x1, f1_val
    return x_base, f_base, trajectory, f.calls


def write_trajectory(filename, trajectory):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("# k x1 x2 F(X)\n")
        for k, (x, fx) in enumerate(trajectory):
            fh.write(f"{k} " + " ".join(f"{v:.12f}" for v in x) + f" {fx:.6e}\n")


def main():
    print(f"Параметри: q = {Q:g}, p = {P:g}, eps1 = {EPS1:g}, eps2 = {EPS2:g}, початковий крок Δx = {STEP0:g}")

    # Тест: функція Розенброка
    x_r, f_r, traj_r, calls_r = hooke_jeeves(rosenbrock, [-1.2, 0.0])
    write_trajectory(out("trajectory_rosenbrock.txt"), traj_r)
    print("\n3. Тест — функція Розенброка, X(0) = (-1.2; 0), точний мінімум (1; 1), f = 0:")
    print(f"   X* = ({x_r[0]:.10f}; {x_r[1]:.10f}), f(X*) = {f_r:.3e}")
    print(f"   кроків траєкторії: {len(traj_r) - 1}, обчислень функції: {calls_r}")
    print(f"   похибка: max|x_i - 1| = {max(abs(x_r[0] - 1), abs(x_r[1] - 1)):.3e}")

    # Система нелінійних рівнянь
    y_exact = (-1 + math.sqrt(13)) / 2
    x_exact = math.sqrt(y_exact + 1)
    print("\n1. Система: x^2 + y^2 - 4 = 0, y - x^2 + 1 = 0")
    print(f"   Точні розв'язки: (±{x_exact:.12f}; {y_exact:.12f}) — з y^2 + y - 3 = 0")
    results = []
    for name, start in (("перший", [1.0, 1.0]), ("другий", [-1.0, 1.0])):
        x, fx, traj, calls = hooke_jeeves(phi, start)
        write_trajectory(out(f"trajectory_system_{len(results) + 1}.txt"), traj)
        err = max(abs(abs(x[0]) - x_exact), abs(x[1] - y_exact))
        results.append((start, x, fx, traj, calls))
        print(f"\n4–5. {name.capitalize()} розв'язок, початкове наближення X(0) = ({start[0]:g}; {start[1]:g}):")
        for k in sorted({0, 1, 2, 3, 5, 10, 20, 40, len(traj) - 1}):
            if k < len(traj):
                (a, b), v = traj[k]
                print(f"   k = {k:3d}: X = ({a: .10f}; {b:.10f}), Φ = {v:.3e}")
        print(f"   Розв'язок X* = ({x[0]:.10f}; {x[1]:.10f}), Φ(X*) = {fx:.3e}")
        print(f"   f1 = {f1(x):.2e}, f2 = {f2(x):.2e}, похибка відносно точного = {err:.2e}")
        print(f"   кроків траєкторії: {len(traj) - 1}, обчислень Φ: {calls}")

    # Графіки
    t = np.linspace(0, 2 * np.pi, 400)
    xs = np.linspace(-2.6, 2.6, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(2 * np.cos(t), 2 * np.sin(t), label="x² + y² − 4 = 0")
    ax1.plot(xs, xs ** 2 - 1, label="y − x² + 1 = 0")
    for start, x, fx, traj, calls in results:
        pts = np.array([p for p, _ in traj])
        ax1.plot(pts[:, 0], pts[:, 1], "k.-", ms=4, lw=0.8)
        ax1.plot(*start, "go")
        ax1.plot(*x, "r*", ms=14)
        ax1.annotate(f"({x[0]:.4f}; {x[1]:.4f})", x, textcoords="offset points", xytext=(-40, 12))
    ax1.set_xlim(-2.6, 2.6)
    ax1.set_ylim(-2.4, 2.6)
    ax1.set_aspect("equal")
    ax1.set_title("Рівняння системи та траєкторії спуску")
    ax1.grid(True)
    ax1.legend(loc="lower right")
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 300), np.linspace(-2.4, 2.6, 300))
    gphi = (gx ** 2 + gy ** 2 - 4) ** 2 + (gy - gx ** 2 + 1) ** 2
    ax2.contour(gx, gy, np.log10(gphi + 1e-12), levels=25, cmap="viridis")
    for start, x, fx, traj, calls in results:
        pts = np.array([p for p, _ in traj])
        ax2.plot(pts[:, 0], pts[:, 1], "r.-", ms=4, lw=0.8)
    ax2.set_aspect("equal")
    ax2.set_title("Лінії рівня lg Φ(X) і траєкторії")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(out("system.png"), dpi=120)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-0.6, 1.6, 300))
    ax1.contour(gx, gy, np.log10(100 * (gx ** 2 - gy) ** 2 + (gx - 1) ** 2 + 1e-12), levels=25, cmap="viridis")
    pts = np.array([p for p, _ in traj_r])
    ax1.plot(pts[:, 0], pts[:, 1], "r.-", ms=3, lw=0.8)
    ax1.plot(1, 1, "k*", ms=14)
    ax1.set_title(f"Функція Розенброка: {len(traj_r) - 1} кроків")
    ax1.grid(True)
    ax2.semilogy([max(v, 1e-30) for _, v in traj_r], label="Розенброк, f(X)")
    for i, (start, x, fx, traj, calls) in enumerate(results, 1):
        ax2.semilogy([max(v, 1e-30) for _, v in traj], label=f"система, розв'язок {i}, Φ(X)")
    ax2.set_xlabel("Крок траєкторії k")
    ax2.set_ylabel("Значення цільової функції")
    ax2.set_title("Спадання цільової функції")
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out("convergence.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
