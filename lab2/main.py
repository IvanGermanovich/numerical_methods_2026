"""Лабораторна робота №2. Розділені різниці.
Інтерполяційні многочлени Ньютона та факторіальні многочлени.

Варіант 2. Планування обчислювальних ресурсів (DevOps):
прогнозування навантаження на CPU за кількістю запитів за секунду (RPS).
"""
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
RPS_FORECAST = 600          # точка прогнозу
NODE_COUNTS = (5, 10, 20)   # кількість вузлів для досліджень
A, B = 50.0, 800.0          # інтервал RPS


def out(name):
    return os.path.join(BASE_DIR, name)


# -------------------------------------------------
# 1. Зчитування та запис даних
# -------------------------------------------------
def read_data(filename):
    """Зчитує експериментальні дані RPS -> CPU (%) з CSV-файлу."""
    x = []
    y = []
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["rps"]))
            y.append(float(row["cpu"]))
    return x, y


def write_nodes(filename, x, y):
    """Записує результати табуляції (вузли x_i, f_i) у текстовий файл."""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("# x_i f_i\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.10f} {yi:.10f}\n")


def read_nodes(filename):
    """Зчитує масиви x_i, f_i, i = 0..n з текстового файлу."""
    x, y = [], []
    with open(filename, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            xi, yi = line.split()
            x.append(float(xi))
            y.append(float(yi))
    return x, y


# -------------------------------------------------
# Функції, що інтерполюються
# -------------------------------------------------
def cpu_model(x):
    """Еталонна модель CPU(RPS), %: базове навантаження 10 %, 0,25 % CPU на кожен
    RPS і нижче завантаження при малій кількості запитів. Проходить через точку
    (50; 20) і відрізняється від інших експериментальних точок не більш ніж на 0,63 %."""
    return 10 + 0.25 * x - 6250 / x**2


def cpu_model_bound(n_nodes, a):
    """M_N / N! для моделі: |f^(N)(x)| = 6250 (N+1)! / x^(N+2), максимум при x = a."""
    return 6250 * (n_nodes + 1) / a ** (n_nodes + 2)


def runge(x):
    """Функція Рунге — класичний приклад розбіжності інтерполяції."""
    return 1 / (1 + 25 * x**2)


# -------------------------------------------------
# 2. Розділені різниці та многочлен Ньютона
# -------------------------------------------------
def omega(x, nodes, k):
    """ω_k(x) = (x - x_0)(x - x_1)...(x - x_k)."""
    result = 1.0
    for i in range(k + 1):
        result = result * (x - nodes[i])
    return result


def divided_difference(x, y, k):
    """Розділена різниця k-го порядку f(x_0, ..., x_k) = Σ f(x_i) / Π (x_i - x_j)."""
    total = 0.0
    for i in range(k + 1):
        denom = 1.0
        for j in range(k + 1):
            if j != i:
                denom *= x[i] - x[j]
        total += y[i] / denom
    return total


def divided_difference_table(x, y):
    """Таблиця розділених різниць: table[k][i] = f(x_i, ..., x_{i+k})."""
    table = [list(y)]
    for k in range(1, len(x)):
        prev = table[-1]
        table.append([(prev[i + 1] - prev[i]) / (x[i + k] - x[i]) for i in range(len(x) - k)])
    return table


def newton_coefficients(x, y):
    """Коефіцієнти многочлена Ньютона f(x_0), f(x_0, x_1), ..., f(x_0, ..., x_n)."""
    return [divided_difference(x, y, k) for k in range(len(x))]


def newton(t, x, coef):
    """N_n(t) = f(x_0) + Σ ω_{k-1}(t) f(x_0, ..., x_k)."""
    result = coef[0]
    for k in range(1, len(coef)):
        result = result + omega(t, x, k - 1) * coef[k]
    return result


def lagrange(t, x, y):
    """Інтерполяційний многочлен Лагранжа (для порівняння)."""
    total = 0.0
    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if j != i:
                term = term * (t - x[j]) / (x[i] - x[j])
        total = total + term
    return total


# -------------------------------------------------
# 3. Скінченні різниці та факторіальні многочлени
# -------------------------------------------------
def finite_difference_table(y):
    """Таблиця скінченних різниць: table[k][i] = Δ^k y_i."""
    table = [list(y)]
    for _ in range(1, len(y)):
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    return table


def finite_differences(y):
    """Скінченні різниці Δ^k f(0), k = 0..n."""
    return [row[0] for row in finite_difference_table(y)]


def factorial_power(t, k):
    """Факторіальний многочлен t^(k) = t(t - 1)...(t - k + 1)."""
    result = 1.0
    for i in range(k):
        result = result * (t - i)
    return result


def factorial_newton(t, diffs):
    """f(t) = Σ Δ^k f(0) / k! · t^(k) — ряд за факторіальними многочленами."""
    total = 0.0
    for k, d in enumerate(diffs):
        total = total + d / math.factorial(k) * factorial_power(t, k)
    return total


# -------------------------------------------------
# Дослідження
# -------------------------------------------------
def forecast_study(x, y):
    """Прогноз CPU(600) за експериментальними даними трьома способами."""
    table = divided_difference_table(x, y)
    coef = newton_coefficients(x, y)
    # Вузли RPS утворюють геометричну прогресію (50, 100, 200, 400, 800), тому
    # заміна t = log2(RPS / 50) дає рівновіддалені вузли t = 0, 1, 2, 3, 4 з кроком 1
    t_nodes = [math.log2(xi / x[0]) for xi in x]
    t_star = math.log2(RPS_FORECAST / x[0])
    # Вплив кількості вузлів: k найближчих до 600 RPS вузлів
    by_nodes = []
    for k in range(2, len(x) + 1):
        idx = sorted(sorted(range(len(x)), key=lambda i: abs(x[i] - RPS_FORECAST))[:k])
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        by_nodes.append((k, xs, newton(RPS_FORECAST, xs, newton_coefficients(xs, ys))))
    return {
        "table": table,
        "coef": coef,
        "coef_check": max(abs(c - table[k][0]) for k, c in enumerate(coef)),
        "newton": newton(RPS_FORECAST, x, coef),
        "lagrange": lagrange(RPS_FORECAST, x, y),
        "t_nodes": t_nodes,
        "t_star": t_star,
        "fin_table": finite_difference_table(y),
        "factorial": factorial_newton(t_star, finite_differences(y)),
        "by_nodes": by_nodes,
    }


def interpolation_study(func, a, b, n_nodes, save=False):
    """Інтерполяція функції func на [a, b] за n_nodes рівновіддаленими вузлами."""
    xs = np.linspace(a, b, n_nodes)
    ys = func(xs)
    if save:  # табуляція у файл і зчитування вхідних даних з файлу
        write_nodes(out(f"nodes_{n_nodes}.txt"), xs, ys)
        xs, ys = read_nodes(out(f"nodes_{n_nodes}.txt"))
    xs = np.array(xs)
    ys = np.array(ys)
    h = (b - a) / (n_nodes - 1)
    coef = newton_coefficients(xs, ys)
    grid = np.linspace(a, b, 20 * (n_nodes - 1) + 1)  # крок h1 = h / 20
    f = func(grid)
    nn = newton(grid, xs, coef)
    eps = np.abs(f - nn)
    w = omega(grid, xs, n_nodes - 1)
    fact = factorial_newton((grid - a) / h, finite_differences(ys))
    lag = lagrange(grid, xs, ys)
    result = {
        "n": n_nodes, "a": a, "b": b, "h": h, "xs": xs, "ys": ys, "grid": grid,
        "f": f, "nn": nn, "eps": eps, "w": w,
        "fact_diff": np.max(np.abs(fact - nn)), "lag_diff": np.max(np.abs(lag - nn)),
    }
    if a <= RPS_FORECAST <= b and func is cpu_model:
        result["at600"] = newton(RPS_FORECAST, xs, coef)
        result["bound"] = np.max(np.abs(w)) * cpu_model_bound(n_nodes, a)
    if save:
        with open(out(f"tabulation_{n_nodes}.txt"), "w", encoding="utf-8") as fh:
            fh.write("# x f(x) N_n(x) eps(x) w_n(x)\n")
            for row in zip(grid, f, nn, eps, w):
                fh.write(" ".join(f"{v:.10e}" for v in row) + "\n")
    return result


def max_error_vs_nodes(func, a, b, counts):
    """Максимальна похибка інтерполяції залежно від кількості вузлів."""
    grid = np.linspace(a, b, 2001)
    errors = []
    for n in counts:
        xs = np.linspace(a, b, n)
        errors.append(np.max(np.abs(func(grid) - newton(grid, xs, newton_coefficients(xs, func(xs))))))
    return errors


def chebyshev_nodes(a, b, n):
    k = np.arange(n)
    return np.sort((a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) * np.pi / (2 * n)))


# -------------------------------------------------
# Вивід і графіки
# -------------------------------------------------
def print_dd_table(x, table):
    print(" i |  x_i  | " + " | ".join(["   f(x_i)   "] + [f" порядок {k}  " for k in range(1, len(x))]))
    for i in range(len(x)):
        cells = [f"{table[k][i]:12.5e}" for k in range(len(x) - i)]
        print(f"{i:2d} | {x[i]:5.0f} | " + " | ".join(cells))


def plot_forecast(x, y, fc):
    xx = np.linspace(A, B, 400)
    diffs = finite_differences(y)
    plt.figure(figsize=(10, 5))
    plt.plot(xx, newton(xx, x, fc["coef"]), label="Многочлен Ньютона N₄(x)")
    plt.plot(xx, factorial_newton(np.log2(xx / x[0]), diffs), "--",
             label="Факторіальний многочлен, t = log₂(RPS/50)")
    plt.plot(x, y, "ko", label="Експериментальні дані")
    plt.plot(RPS_FORECAST, fc["newton"], "r*", ms=14, label=f"Прогноз (Ньютон): {fc['newton']:.2f} %")
    plt.plot(RPS_FORECAST, fc["factorial"], "g*", ms=14,
             label=f"Прогноз (факторіальні многочлени): {fc['factorial']:.2f} %")
    plt.xlabel("Навантаження, RPS")
    plt.ylabel("Завантаження CPU, %")
    plt.title("Модель CPU = f(RPS) і прогноз для 600 RPS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("forecast.png"), dpi=120)


def plot_fixed_interval(studies):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(studies[-1]["grid"], studies[-1]["f"], "k", lw=3, alpha=0.4, label="f(x) — еталонна модель")
    for s in studies:
        line, = ax1.plot(s["grid"], s["nn"], "--", label=f"N(x), {s['n']} вузлів")
        ax1.plot(s["xs"], s["ys"], "o", ms=4, color=line.get_color())
        ax2.semilogy(s["grid"], np.maximum(s["eps"], 1e-16), color=line.get_color(),
                     label=f"ε(x), {s['n']} вузлів")
    ax1.set_ylabel("CPU, %")
    ax1.set_title("Інтерполяція моделі на [50; 800] (фіксований інтервал, різний крок)")
    ax1.grid(True)
    ax1.legend()
    ax2.set_xlabel("Навантаження, RPS")
    ax2.set_ylabel("ε(x) = |f(x) − N(x)|, %")
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out("fixed_interval.png"), dpi=120)


def plot_omega(studies):
    plt.figure(figsize=(10, 4.5))
    for s in studies:
        plt.plot(s["grid"], s["w"] / np.max(np.abs(s["w"])), label=f"ωₙ(x) / max|ωₙ|, {s['n']} вузлів")
    plt.xlabel("Навантаження, RPS")
    plt.ylabel("Нормоване ωₙ(x)")
    plt.title("Функція ωₙ(x) = (x − x₀)(x − x₁)…(x − xₙ)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("omega.png"), dpi=120)


def plot_fixed_step(studies):
    plt.figure(figsize=(10, 4.5))
    for s in studies:
        plt.semilogy(s["grid"], np.maximum(s["eps"], 1e-16),
                     label=f"{s['n']} вузлів, [50; {s['b']:.1f}]")
    plt.xlabel("Навантаження, RPS")
    plt.ylabel("ε(x), %")
    plt.title(f"Фіксований крок h = {studies[0]['h']:.2f}, різний інтервал")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out("fixed_step.png"), dpi=120)


def plot_runge(studies, cheb, counts, err_model, err_runge):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    xx = np.linspace(-1, 1, 801)
    ax1.plot(xx, runge(xx), "k", lw=2.5, label="f(x) = 1/(1 + 25x²)")
    for s in studies:
        ax1.plot(s["grid"], s["nn"], label=f"{s['n']} рівновіддалених вузлів")
    ax1.plot(xx, newton(xx, cheb, newton_coefficients(cheb, runge(cheb))), "k:", label="20 вузлів Чебишова")
    ax1.set_ylim(-1.5, 2)
    ax1.set_title("Ефект Рунге")
    ax1.grid(True)
    ax1.legend(fontsize=8)
    ax2.semilogy(counts, err_model, "o-", label="модель CPU на [50; 800]")
    ax2.semilogy(counts, err_runge, "s-", label="функція Рунге на [−1; 1]")
    ax2.set_xlabel("Кількість вузлів")
    ax2.set_ylabel("max ε(x)")
    ax2.set_title("Максимальна похибка залежно від кількості вузлів")
    ax2.grid(True, which="both", alpha=0.4)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out("runge.png"), dpi=120)


def main():
    # 1. Зчитування експериментальних даних з CSV-файлу
    x, y = read_data(DATA_FILE)
    print("Експериментальні дані:")
    print("  RPS  | CPU, %")
    for xi, yi in zip(x, y):
        print(f"{xi:6.0f} | {yi:6.1f}")

    # 2–3. Таблиця розділених різниць і прогноз для 600 RPS
    fc = forecast_study(x, y)
    print("\nТаблиця розділених різниць:")
    print_dd_table(x, fc["table"])
    print("\nКоефіцієнти многочлена Ньютона f(x_0, ..., x_k):")
    for k, c in enumerate(fc["coef"]):
        print(f"  k = {k}: {c: .10e}")
    print(f"Перевірка (явна формула / рекурентна таблиця): max різниця = {fc['coef_check']:.2e}")
    print(f"\nПрогноз CPU({RPS_FORECAST}) — многочлен Ньютона:  {fc['newton']:.4f} %")
    print(f"Прогноз CPU({RPS_FORECAST}) — многочлен Лагранжа: {fc['lagrange']:.4f} %"
          f" (різниця {abs(fc['newton'] - fc['lagrange']):.1e})")

    print("\nФакторіальні многочлени: заміна t = log2(RPS / 50) дає вузли t =",
          ", ".join(f"{t:g}" for t in fc["t_nodes"]), "(крок 1)")
    print("Таблиця скінченних різниць:")
    for k, row in enumerate(fc["fin_table"]):
        print(f"  Δ^{k}: " + "  ".join(f"{v:7.2f}" for v in row))
    print(f"t* = log2({RPS_FORECAST} / 50) = {fc['t_star']:.5f}")
    print(f"Прогноз CPU({RPS_FORECAST}) — факторіальні многочлени: {fc['factorial']:.4f} %")

    print(f"\nВплив кількості вузлів на прогноз (k найближчих до {RPS_FORECAST} RPS вузлів):")
    for k, xs, val in fc["by_nodes"]:
        print(f"  k = {k}: вузли {', '.join(f'{v:.0f}' for v in xs):22s} -> CPU = {val:.4f} %")

    # 4. Графік CPU(RPS)
    plot_forecast(x, y, fc)

    # 5. Дослідження точності на еталонній моделі
    print("\nЕталонна модель f(x) = 10 + 0.25x − 6250/x²")
    for xi, yi in zip(x, y):
        print(f"  f({xi:.0f}) = {cpu_model(xi):8.4f}  (дані: {yi:.1f}, відхилення {cpu_model(xi) - yi:+.4f})")
    print(f"  f({RPS_FORECAST}) = {cpu_model(RPS_FORECAST):.4f} %")

    fixed = [interpolation_study(cpu_model, A, B, n, save=True) for n in NODE_COUNTS]
    print(f"\nДослідження 1. Фіксований інтервал [{A:.0f}; {B:.0f}], різний крок:")
    print("Вузлів |   h    |  max ε(x)  | середня ε  | ε(600)     | теор. оцінка | факт.−Ньютон | Лагранж−Ньютон")
    for s in fixed:
        print(f"{s['n']:6d} | {s['h']:6.2f} | {np.max(s['eps']):10.3e} | {np.mean(s['eps']):10.3e} | "
              f"{abs(s['at600'] - cpu_model(RPS_FORECAST)):10.3e} | {s['bound']:12.3e} | "
              f"{s['fact_diff']:12.1e} | {s['lag_diff']:10.1e}")
    for s in fixed:
        print(f"  {s['n']} вузлів: прогноз CPU({RPS_FORECAST}) = {s['at600']:.6f} %")
    plot_fixed_interval(fixed)
    plot_omega(fixed)

    h = (B - A) / (NODE_COUNTS[-1] - 1)
    step = [interpolation_study(cpu_model, A, A + h * (n - 1), n) for n in NODE_COUNTS]
    print(f"\nДослідження 2. Фіксований крок h = {h:.4f}, інтервал [50; 50 + (n − 1)h]:")
    print("Вузлів |    b     |  max ε(x)  | середня ε")
    for s in step:
        print(f"{s['n']:6d} | {s['b']:8.2f} | {np.max(s['eps']):10.3e} | {np.mean(s['eps']):10.3e}")
    plot_fixed_step(step)

    # Ефект Рунге
    runge_studies = [interpolation_study(runge, -1.0, 1.0, n) for n in NODE_COUNTS]
    cheb = chebyshev_nodes(-1.0, 1.0, 20)
    grid = np.linspace(-1, 1, 2001)
    cheb_err = np.max(np.abs(runge(grid) - newton(grid, cheb, newton_coefficients(cheb, runge(cheb)))))
    counts = list(range(3, 31))
    err_model = max_error_vs_nodes(cpu_model, A, B, counts)
    err_runge = max_error_vs_nodes(runge, -1.0, 1.0, counts)
    print("\nДослідження 3. Ефект Рунге, f(x) = 1/(1 + 25x²) на [−1; 1]:")
    for s in runge_studies:
        print(f"  {s['n']:2d} рівновіддалених вузлів: max ε = {np.max(s['eps']):.4f}")
    print(f"  20 вузлів Чебишова:         max ε = {cheb_err:.4f}")
    print("Максимальна похибка залежно від кількості вузлів (модель CPU / функція Рунге):")
    for n, em, er in zip(counts, err_model, err_runge):
        print(f"  {n:2d}: {em:10.3e} / {er:10.3e}")
    plot_runge(runge_studies, cheb, counts, err_model, err_runge)

    plt.show()


if __name__ == "__main__":
    main()
