import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import numpy as np


def analytic(matrix):
    C = np.array(matrix, dtype=float)

    n, m = C.shape
    if n != m:
        raise ValueError("Для метода обратной матрицы нужна квадратная матрица.")

    det = np.linalg.det(C)
    if abs(det) < 1e-12:
        raise ValueError("Матрица вырождена, аналитический метод в таком виде неприменим.")

    C_inv = np.linalg.inv(C)
    u = np.ones((n, 1))

    denominator = float((u.T @ C_inv @ u)[0, 0])
    if abs(denominator) < 1e-12:
        raise ValueError("Знаменатель равен нулю, цена игры не может быть вычислена этим способом.")

    x_opt = (C_inv @ u / denominator).flatten()
    y_opt = (u.T @ C_inv / denominator).flatten()
    v = 1 / denominator

    return x_opt, y_opt, v


def brown_robinson(M, eps_target=0.1, max_iter=10000):
    C = np.array(M, dtype=float)
    m, n = C.shape

    a_counts = np.zeros(m, dtype=int)
    b_counts = np.zeros(n, dtype=int)

    i = random.randrange(0, n)
    j = random.randrange(0, m)
    a_counts[i] += 1
    b_counts[j] += 1

    row_sums = C[:, j].copy()
    col_sums = C[i, :].copy()

    history = []

    for k in range(1, max_iter + 1):
        upper_value = row_sums.max() / k
        lower_value = col_sums.min() / k
        eps = upper_value - lower_value

        history.append({
            "k": k,
            "A_choice": i + 1,
            "B_choice": j + 1,
            "row_sum_1": row_sums[0],
            "row_sum_2": row_sums[1],
            "row_sum_3": row_sums[2],
            "col_sum_1": col_sums[0],
            "col_sum_2": col_sums[1],
            "col_sum_3": col_sums[2],
            "upper_value": upper_value,
            "lower_value": lower_value,
            "epsilon": eps,
            "x1": a_counts[0],
            "x2": a_counts[1],
            "x3": a_counts[2],
            "y1": b_counts[0],
            "y2": b_counts[1],
            "y3": b_counts[2],
        })

        if eps <= eps_target:
            break

        next_i = np.argmax(row_sums)

        next_j = np.argmin(col_sums)

        i, j = next_i, next_j

        a_counts[i] += 1
        b_counts[j] += 1

        row_sums += C[:, j]
        col_sums += C[i, :]

    history_df = pd.DataFrame(history)

    k_final = len(history_df)
    p_est = a_counts / k_final
    q_est = b_counts / k_final

    game_value_interval = (
        history_df.iloc[-1]["lower_value"],
        history_df.iloc[-1]["upper_value"]
    )

    return history_df, p_est, q_est, game_value_interval, k_final


if __name__ == "__main__":
    C = [
        [13,  2,  4],
        [ 7,  6, 10],
        [ 8, 14,  6]
    ]
    
    x_opt, y_opt, v = analytic(C)

    print("Матрица игры:")
    print(np.array(C))
    print()


    print("Аналитическое решение:")
    print("Оптимальная смешанная стратегия игрока A:")
    print(x_opt)
    print()

    print("Оптимальная смешанная стратегия игрока B:")
    print(y_opt)
    print()

    print("Цена игры:")
    print(v)
    print()
    print()

    
    eps_target = 0.1

    history_df, p_est, q_est, value_interval, k_final = brown_robinson(C, eps_target=eps_target)

    print("Метод Брауна–Робинсона:")
    print(f"Количество итераций: {k_final}")
    print(f"Cтратегия игрока A: p = {np.round(q_est, 4)}")
    print(f"Cтратегия игрока B: q = {np.round(p_est, 4)}")
    print(f"Оценка цены игры: {(value_interval[0] + value_interval[1]) / 2:.4f}")
    print()

    print("Первые 6 строк таблицы:")
    print(history_df.head(6).to_string(index=False))
    print()

    print("Последние 6 строк таблицы:")
    print(history_df.tail(6).to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["k"], history_df["epsilon"], label="Ошибка ε[k]")
    plt.axhline(y=0.1, linestyle="--", label="Требуемая точность ε = 0.1")

    plt.xlabel("Номер итерации")
    plt.ylabel("Ошибка")
    plt.title("Убывание ошибки в методе Брауна–Робинсона")
    plt.grid()
    plt.legend()
    plt.show()