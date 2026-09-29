"""Лабораторна робота №3. Знаходження алгебраїчних многочленів найкращого
квадратичного наближення методом найменших квадратів.

Апроксимація середньомісячних температур за 24 місяці та прогноз на наступні 3 місяці.
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
NODES_FILE = os.path.join(BASE_DIR, "nodes.txt")
MAX_DEGREE = 10
FUTURE = np.array([25.0, 26.0, 27.0])  # прогноз на наступні 3 місяці


def out(name):
    return os.path.join(BASE_DIR, name)


# -------------------------------------------------
# 1. Вхідні дані
# -------------------------------------------------
def read_csv(filename):
    """Зчитує середньомісячні температури з CSV-файлу."""
    x, y = [], []
    with open(filename, newline="") as fh:
        for row in csv.DictReader(fh):
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))
    return np.array(x), np.array(y)


def write_nodes(filename, x, y):
    """Записує результати табуляції (вузли x_i, f_i) у текстовий файл."""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("# x_i f_i\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:g} {yi:g}\n")


def read_nodes(filename):
    """Зчитує масиви x_i, f_i, i = 0..n з текстового файлу."""
    data = np.loadtxt(filename, comments="#")
    return data[:, 0], data[:, 1]


# -------------------------------------------------
# 2. Метод найменших квадратів
# -------------------------------------------------
def form_matrix(t, m, rho):
    """Масив B: b_kl = Σ ρ_i t_i^(k+l), k, l = 0..m."""
    B = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for l in range(m + 1):
            B[k, l] = np.sum(rho * t ** (k + l))
    return B


def form_vector(t, f, m, rho):
    """Масив C: c_k = Σ ρ_i f_i t_i^k, k = 0..m."""
    return np.array([np.sum(rho * f * t ** k) for k in range(m + 1)])


def gauss_solve(A, b):
    """Розв'язує A x = b методом Гауса з вибором головного елемента по стовпцях."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    # прямий хід
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))  # рядок з найбільшим |a_ik|
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    # зворотний хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def polynomial(t, coef):
    """φ(t) = a_0 + a_1 t + ... + a_m t^m (схема Горнера)."""
    result = np.zeros_like(np.asarray(t, dtype=float))
    for a in coef[::-1]:
        result = result * t + a
    return result


def dispersion(f, phi):
    """δ = sqrt(Σ (φ(x_i) - f_i)^2 / (n + 1))."""
    return np.sqrt(np.sum((phi - f) ** 2) / len(f))


def error(f, phi):
    """ε(x) = |f(x) - φ(x)|."""
    return np.abs(f - phi)


def fit(x, f, m, rho=None):
    """Многочлен найкращого квадратичного наближення степеня m.

    Щоб зменшити похибки округлення, аргумент нормується: t = (x - c) / s ∈ [-1; 1].
    Без нормування число обумовленості B для m = 10 сягає 10^28 і метод Гауса дає хибний результат.
    """
    rho = np.ones_like(f) if rho is None else rho  # ваги ρ_i = 1 (точність усіх значень однакова)
    c = (x[0] + x[-1]) / 2
    s = (x[-1] - x[0]) / 2
    t = (x - c) / s
    B = form_matrix(t, m, rho)
    C = form_vector(t, f, m, rho)
    coef = gauss_solve(B, C)
    return {"m": m, "coef": coef, "c": c, "s": s, "cond": np.linalg.cond(B),
            "residual": np.max(np.abs(B @ coef - C))}


def evaluate(model, x):
    """Значення многочлена φ(x) за масивом коефіцієнтів."""
    return polynomial((np.asarray(x, dtype=float) - model["c"]) / model["s"], model["coef"])


def raw_condition(x, m):
    """Число обумовленості B без нормування аргументу (b_kl = Σ x_i^(k+l))."""
    B = np.array([[np.sum(x ** (k + l)) for l in range(m + 1)] for k in range(m + 1)])
    return np.linalg.cond(B)


# Додатково: той самий МНК з тригонометричним базисом 1, cos(2πx/12), sin(2πx/12)
def fit_harmonic(x, f, period=12.0):
    basis = np.vstack([np.ones_like(x), np.cos(2 * np.pi * x / period), np.sin(2 * np.pi * x / period)])
    return gauss_solve(basis @ basis.T, basis @ f)


def eval_harmonic(coef, x, period=12.0):
    x = np.asarray(x, dtype=float)
    return coef[0] + coef[1] * np.cos(2 * np.pi * x / period) + coef[2] * np.sin(2 * np.pi * x / period)


# -------------------------------------------------
# Графіки
# -------------------------------------------------
def plot_dispersion(models, deltas, m_opt):
    degrees = [md["m"] for md in models]
    plt.figure(figsize=(10, 4.5))
    plt.plot(degrees, deltas, "o-")
    plt.plot(m_opt, deltas[m_opt - 1], "r*", ms=16, label=f"мінімум: m = {m_opt}, δ = {deltas[m_opt - 1]:.3f} °C")
    plt.xticks(degrees)
    plt.xlabel("Степінь многочлена m")
    plt.ylabel("Дисперсія δ, °C")
    plt.title("Залежність дисперсії від степеня апроксимуючого многочлена")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("dispersion.png"), dpi=120)


def plot_approximation(x, f, models, m_opt, harm):
    xx = np.linspace(x[0], FUTURE[-1], 600)
    plt.figure(figsize=(10, 5.5))
    plt.plot(x, f, "ko", label="Фактичні дані")
    for m in (2, 4, 6):
        plt.plot(xx, evaluate(models[m - 1], xx), lw=1, alpha=0.7, label=f"φ(x), m = {m}")
    best = models[m_opt - 1]
    plt.plot(xx, evaluate(best, xx), "r", lw=2.2, label=f"φ(x), m = {m_opt} (оптимальний)")
    plt.plot(xx, eval_harmonic(harm, xx), "g--", lw=1.8, label="Тригонометричний МНК (додатково)")
    plt.plot(FUTURE, eval_harmonic(harm, FUTURE), "g^", ms=9)
    plt.axvspan(x[-1], FUTURE[-1], color="grey", alpha=0.15, label="Прогноз (місяці 25–27)")
    plt.ylim(-20, 40)
    plt.xlabel("Місяць")
    plt.ylabel("Температура, °C")
    plt.title("Апроксимація середньомісячних температур методом найменших квадратів")
    plt.grid(True)
    plt.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out("approximation.png"), dpi=120)


def plot_errors(grid, errors):
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True, sharey=True)
    for ax, (m, e) in zip(axes.flat, errors.items()):
        ax.plot(grid, e, lw=1)
        ax.set_title(f"m = {m}")
        ax.grid(True)
    for ax in axes[1]:
        ax.set_xlabel("Місяць")
    for ax in axes[:, 0]:
        ax.set_ylabel("ε(x), °C")
    fig.suptitle("Похибка апроксимації ε(x) = |f(x) − φ(x)| для m = 1…10")
    fig.tight_layout()
    fig.savefig(out("errors.png"), dpi=120)


def main():
    # 1. Табуляція даних і зчитування вузлів з текстового файлу
    x_csv, f_csv = read_csv(DATA_FILE)
    n = len(x_csv)
    h = (x_csv[-1] - x_csv[0]) / (n - 1)
    write_nodes(NODES_FILE, x_csv, f_csv)
    x, f = read_nodes(NODES_FILE)
    print(f"Кількість вузлів: {n}, відрізок [{x[0]:g}; {x[-1]:g}], крок h = {h:g} міс.")
    print(" x_i | f_i, °C")
    for xi, fi in zip(x, f):
        print(f"{xi:4.0f} | {fi:5.1f}")

    # 2–3. Многочлени степенів m = 1..10 та дисперсія
    models = [fit(x, f, m) for m in range(1, MAX_DEGREE + 1)]
    deltas = [dispersion(f, evaluate(md, x)) for md in models]
    m_opt = int(np.argmin(deltas)) + 1
    print("\n m | дисперсія δ, °C | cond(B), x | cond(B), t | max|B·a − C| | прогноз 25, 26, 27")
    for md, d in zip(models, deltas):
        fut = evaluate(md, FUTURE)
        print(f"{md['m']:2d} | {d:15.4f} | {raw_condition(x, md['m']):10.1e} | {md['cond']:10.1e} | "
              f"{md['residual']:12.1e} | " + ", ".join(f"{v:8.2f}" for v in fut))
    print(f"\nОптимальний степінь за мінімумом дисперсії: m = {m_opt} (δ = {deltas[m_opt - 1]:.4f} °C)")

    best = models[m_opt - 1]
    print(f"Коефіцієнти многочлена m = {m_opt} (змінна t = (x − {best['c']:g}) / {best['s']:g}):")
    for j, a in enumerate(best["coef"]):
        print(f"  a_{j} = {a: .6f}")

    # 4. Табуляція похибки з кроком h1 = (x_n - x_0) / (20 n); між вузлами f(x) — лінійна інтерполяція
    grid = np.linspace(x[0], x[-1], 20 * (n - 1) + 1)
    f_grid = np.interp(grid, x, f)
    errors = {md["m"]: error(f_grid, evaluate(md, grid)) for md in models}
    with open(out("errors.txt"), "w", encoding="utf-8") as fh:
        fh.write("# x f(x) " + " ".join(f"eps_m{m}" for m in errors) + "\n")
        for k, xv in enumerate(grid):
            fh.write(f"{xv:.2f} {f_grid[k]:.3f} " + " ".join(f"{errors[m][k]:.4f}" for m in errors) + "\n")
    print("\nПохибка у вузлах для оптимального многочлена:")
    node_err = error(f, evaluate(best, x))
    print(f"  max ε = {node_err.max():.4f} °C (місяць {x[np.argmax(node_err)]:g}), середня ε = {node_err.mean():.4f} °C")

    # 5. Прогноз на наступні 3 місяці
    harm = fit_harmonic(x, f)
    harm_delta = dispersion(f, eval_harmonic(harm, x))
    print(f"\nПрогноз (m = {m_opt}): " + ", ".join(f"місяць {m:g}: {v:.2f} °C" for m, v in zip(FUTURE, evaluate(best, FUTURE))))
    print(f"Тригонометричний МНК: φ(x) = {harm[0]:.4f} {harm[1]:+.4f}·cos(2πx/12) {harm[2]:+.4f}·sin(2πx/12),"
          f" δ = {harm_delta:.4f} °C")
    print("Прогноз (тригонометричний МНК): " + ", ".join(f"місяць {m:g}: {v:.2f} °C" for m, v in zip(FUTURE, eval_harmonic(harm, FUTURE))))

    plot_dispersion(models, deltas, m_opt)
    plot_approximation(x, f, models, m_opt, harm)
    plot_errors(grid, errors)
    plt.show()


if __name__ == "__main__":
    main()
