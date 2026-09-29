"""Лабораторна робота 0 (вступна).
Організація робочого середовища та основи роботи з Python і Git.
"""
import numpy as np


def f(x):
    return x**2 - 4*x + 3


def main():
    print("Hello, world")

    # 1. Змінні та типи даних
    a = 5
    x = 3.14
    name = "Numerical Methods"
    print(f"a = {a} ({type(a).__name__}), x = {x} ({type(x).__name__}), name = {name!r}")

    # 2. Масиви NumPy
    A = np.array([[1, 2],
                  [3, 4]])
    b = np.array([5, 6])
    print("A =\n", A)
    print("b =", b)
    print("A @ b =", A @ b)
    print("Розв'язок A x = b:", np.linalg.solve(A, b))

    # 3. Функції
    print("f(1) =", f(1), " f(3) =", f(3))

    # 4. Цикли
    for i in range(5):
        print(i)

    # 5. Умовні оператори
    if x > 0:
        print("Positive")
    else:
        print("Non-positive")


if __name__ == "__main__":
    main()
