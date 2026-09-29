"""Лабораторна робота №1. Інтерполяція кубічними сплайнами.

Профіль висоти маршруту від станції Заросляк до вершини Говерли.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "route_data.json")
TAB_FILE = os.path.join(BASE_DIR, "tabulation.txt")

LOCATIONS = (
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)
URL = "https://api.open-elevation.com/api/v1/lookup?locations=" + LOCATIONS


# -------------------------------
# 1. Запит до Open-Elevation API
# -------------------------------
def fetch_route():
    """Отримує дані з API; якщо API недоступне — читає збережену копію."""
    try:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        with open(DATA_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except (requests.RequestException, ValueError) as exc:
        print("API недоступне (", exc, "), використовуємо", DATA_FILE)
        with open(DATA_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    return data["results"]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ----------------------------------------------
# 6. Коефіцієнти тридіагональної системи для c_i
# ----------------------------------------------
def spline_system(x, y):
    """Формує тридіагональну систему для коефіцієнтів c_2..c_n.

    Сплайн на [x_{i-1}, x_i], i = 1..n:
        S_i(x) = a_i + b_i (x - x_{i-1}) + c_i (x - x_{i-1})^2 + d_i (x - x_{i-1})^3
    Вільний сплайн: c_1 = 0, c_{n+1} = 0.
    Рівняння для i = 2..n:
        h_{i-1} c_{i-1} + 2 (h_{i-1} + h_i) c_i + h_i c_{i+1}
            = 3 ((y_i - y_{i-1}) / h_i - (y_{i-1} - y_{i-2}) / h_{i-1})
    Повертає масиви alpha (піддіагональ), beta (діагональ),
    gamma (наддіагональ), delta (вільні члени), довжиною n - 1.
    """
    h = np.diff(x)          # h[k] = h_{k+1}, k = 0..n-1
    n = len(h)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)
    gamma = np.zeros(n - 1)
    delta = np.zeros(n - 1)
    for k in range(n - 1):  # k-те рівняння відповідає c_{k+2}
        alpha[k] = h[k] if k > 0 else 0.0
        beta[k] = 2 * (h[k] + h[k + 1])
        gamma[k] = h[k + 1] if k < n - 2 else 0.0
        delta[k] = 3 * ((y[k + 2] - y[k + 1]) / h[k + 1] - (y[k + 1] - y[k]) / h[k])
    return alpha, beta, gamma, delta


# ------------------------------------------
# 7. Метод прогонки (алгоритм Томаса)
# ------------------------------------------
def tridiagonal_solve(alpha, beta, gamma, delta):
    """Розв'язує alpha_i x_{i-1} + beta_i x_i + gamma_i x_{i+1} = delta_i."""
    m = len(beta)
    A = np.zeros(m)  # прогонкові коефіцієнти
    B = np.zeros(m)
    # пряма прогонка
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, m):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom
    # зворотна прогонка
    sol = np.zeros(m)
    sol[-1] = B[-1]
    for i in range(m - 2, -1, -1):
        sol[i] = A[i] * sol[i + 1] + B[i]
    return sol


def tridiagonal_residual(alpha, beta, gamma, delta, sol):
    """Перевірка: обчислює ліву частину системи і порівнює з delta."""
    lhs = beta * sol
    lhs[1:] += alpha[1:] * sol[:-1]
    lhs[:-1] += gamma[:-1] * sol[1:]
    return np.max(np.abs(lhs - delta))


# ---------------------------------------
# 8–9. Коефіцієнти a_i, b_i, c_i, d_i
# ---------------------------------------
def cubic_spline(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    n = len(h)
    alpha, beta, gamma, delta = spline_system(x, y)
    c_inner = tridiagonal_solve(alpha, beta, gamma, delta)
    c_all = np.concatenate(([0.0], c_inner, [0.0]))  # c_1..c_{n+1}
    residual = tridiagonal_residual(alpha, beta, gamma, delta, c_inner)

    a = y[:-1].copy()
    c = c_all[:-1]
    d = (c_all[1:] - c_all[:-1]) / (3 * h)
    b = (y[1:] - y[:-1]) / h - h * (c_all[1:] + 2 * c_all[:-1]) / 3
    return {
        "x": x, "a": a, "b": b, "c": c, "d": d,
        "system": (alpha, beta, gamma, delta), "residual": residual,
    }


def spline_eval(sp, xx):
    x = sp["x"]
    idx = np.clip(np.searchsorted(x, xx, side="right") - 1, 0, len(x) - 2)
    t = xx - x[idx]
    return sp["a"][idx] + sp["b"][idx] * t + sp["c"][idx] * t**2 + sp["d"][idx] * t**3


def spline_derivative(sp, xx):
    x = sp["x"]
    idx = np.clip(np.searchsorted(x, xx, side="right") - 1, 0, len(x) - 2)
    t = xx - x[idx]
    return sp["b"][idx] + 2 * sp["c"][idx] * t + 3 * sp["d"][idx] * t**2


def print_spline(sp, title):
    alpha, beta, gamma, delta = sp["system"]
    print(f"\n===== {title} =====")
    print("Коефіцієнти тридіагональної системи (для c_2..c_n):")
    print(" i |     alpha     |     beta      |     gamma     |     delta")
    for k in range(len(beta)):
        print(f"{k + 2:2d} | {alpha[k]:13.4f} | {beta[k]:13.4f} | "
              f"{gamma[k]:13.4f} | {delta[k]:13.6f}")
    print(f"Перевірка прогонки: max|Ac - delta| = {sp['residual']:.3e}")
    print("\nКоефіцієнти сплайнів:")
    print(" i |     a_i      |     b_i      |      c_i       |      d_i")
    for i in range(len(sp["a"])):
        print(f"{i + 1:2d} | {sp['a'][i]:12.4f} | {sp['b'][i]:12.6f} | "
              f"{sp['c'][i]:14.8f} | {sp['d'][i]:14.10f}")


def main():
    # 1–3. Отримання даних і табуляція
    results = fetch_route()
    n = len(results)
    lines = [f"Кількість вузлів: {n}", "", "Табуляція вузлів:",
             "№  | Latitude  | Longitude | Elevation (m)"]
    for i, p in enumerate(results):
        lines.append(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | "
                     f"{p['elevation']:.2f}")

    # 4. Кумулятивна відстань
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = np.array([p["elevation"] for p in results])
    distances = [0.0]
    for i in range(1, n):
        distances.append(distances[-1] + haversine(*coords[i - 1], *coords[i]))
    distances = np.array(distances)

    lines += ["", "Табуляція (відстань, висота):", "№  | Distance (m) | Elevation (m)"]
    for i in range(n):
        lines.append(f"{i:2d} | {distances[i]:12.2f} | {elevations[i]:10.2f}")
    text = "\n".join(lines)
    print(text)
    with open(TAB_FILE, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nТабуляцію записано у файл {os.path.basename(TAB_FILE)}")

    # 5. Графік дискретних даних
    plt.figure(figsize=(10, 5))
    plt.plot(distances, elevations, "o-", label="GPS-вузли")
    plt.xlabel("Кумулятивна відстань, м")
    plt.ylabel("Висота, м")
    plt.title("Профіль висоти: Заросляк → Говерла (дискретні дані)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "profile_points.png"), dpi=120)

    # 6–9. Сплайн за всіма вузлами (еталонна функція f(x))
    sp_full = cubic_spline(distances, elevations)
    print_spline(sp_full, f"Кубічний сплайн, {n} вузлів")

    xx = np.linspace(distances[0], distances[-1], 2000)
    yy_full = spline_eval(sp_full, xx)

    # 10–12. Сплайни з 10, 15, 20 вузлами
    node_counts = [10, 15, 20]
    fig, axes = plt.subplots(len(node_counts), 1, figsize=(10, 12), sharex=True)
    fig_err, ax_err = plt.subplots(figsize=(10, 5))
    accuracy = []
    for ax, k in zip(axes, node_counts):
        idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
        sp_k = cubic_spline(distances[idx], elevations[idx])
        print_spline(sp_k, f"Кубічний сплайн, {k} вузлів")
        yy_k = spline_eval(sp_k, xx)
        err = np.abs(yy_full - yy_k)
        err_nodes = np.abs(elevations - spline_eval(sp_k, distances))
        accuracy.append((k, err.max(), err.mean(), err_nodes.max()))

        ax.plot(xx, yy_full, "k--", lw=1, label=f"f(x): сплайн по {n} вузлах")
        ax.plot(xx, yy_k, lw=2, label=f"φ(x): сплайн по {k} вузлах")
        ax.plot(distances[idx], elevations[idx], "ro", label="вузли інтерполяції")
        ax.plot(distances, elevations, "k.", ms=4, label="всі GPS-точки")
        ax.set_ylabel("Висота, м")
        ax.set_title(f"{k} вузлів")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax_err.plot(xx, err, label=f"ε(x), {k} вузлів")

    print("\n===== Вплив кількості вузлів на точність =====")
    print("Вузлів | max|f - φ| (м) | середня |f - φ| (м) | max похибка у GPS-вузлах (м)")
    for k, e_max, e_mean, e_nodes in accuracy:
        print(f"{k:6d} | {e_max:14.3f} | {e_mean:19.3f} | {e_nodes:28.3f}")

    axes[-1].set_xlabel("Кумулятивна відстань, м")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "spline_nodes.png"), dpi=120)

    ax_err.set_xlabel("Кумулятивна відстань, м")
    ax_err.set_ylabel("|f(x) - φ(x)|, м")
    ax_err.set_title("Похибка інтерполяції ε(x) = |f(x) - φ(x)|")
    ax_err.grid(True)
    ax_err.legend()
    fig_err.tight_layout()
    fig_err.savefig(os.path.join(BASE_DIR, "spline_error.png"), dpi=120)

    # Додатково 1. Характеристики маршруту
    print("\n===== Характеристики маршруту =====")
    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
    print("Загальна довжина маршруту (м):", round(distances[-1], 2))
    print("Сумарний набір висоти (м):", round(total_ascent, 2))
    print("Сумарний спуск (м):", round(total_descent, 2))

    # Додатково 2. Аналіз градієнта через похідну сплайна
    grad_full = spline_derivative(sp_full, xx) * 100
    print("\n===== Аналіз градієнта =====")
    print("Максимальний підйом (%):", round(np.max(grad_full), 2))
    print("Максимальний спуск (%):", round(np.min(grad_full), 2))
    print("Середній градієнт (%):", round(np.mean(np.abs(grad_full)), 2))
    steep = np.abs(grad_full) > 15
    print(f"Частка маршруту з крутизною > 15%: {steep.mean() * 100:.1f}%")
    # межі ділянок з крутизною > 15%
    edges = np.flatnonzero(np.diff(steep.astype(int)))
    starts = list(xx[edges[np.diff(steep.astype(int))[edges] == 1] + 1])
    ends = list(xx[edges[np.diff(steep.astype(int))[edges] == -1]])
    if steep[0]:
        starts.insert(0, xx[0])
    if steep[-1]:
        ends.append(xx[-1])
    for s, e in zip(starts, ends):
        print(f"  ділянка {s:7.1f} – {e:7.1f} м")

    plt.figure(figsize=(10, 5))
    plt.plot(xx, grad_full)
    plt.axhline(15, color="r", ls="--", lw=1, label="±15%")
    plt.axhline(-15, color="r", ls="--", lw=1)
    plt.xlabel("Кумулятивна відстань, м")
    plt.ylabel("Градієнт, %")
    plt.title("Крутизна маршруту (похідна сплайна)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "gradient.png"), dpi=120)

    # Додатково 3. Механічна енергія підйому
    mass = 80
    g = 9.81
    energy = mass * g * total_ascent
    print("\n===== Механічна енергія підйому (m = 80 кг) =====")
    print("Механічна робота (Дж):", round(energy, 2))
    print("Механічна робота (кДж):", round(energy / 1000, 2))
    print("Енергія (ккал):", round(energy / 4184, 2))

    plt.show()


if __name__ == "__main__":
    main()
