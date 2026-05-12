from __future__ import annotations
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, Rational

class Bimatrix:
    def __init__(self, name: str, A: NDArray[np.integer], B: NDArray[np.integer]) -> None:
        self.name = name
        self.A = A 
        self.B = B 


def matrix(values: list[list[int]]) -> NDArray[np.integer]:
    return np.array(values, dtype=np.int64)


def one_based(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(i + 1, j + 1) for i, j in cells]


def find_nash(A: NDArray[np.integer], B: NDArray[np.integer]) -> list[tuple[int, int]]:
    row_best_for_player_1 = A == A.max(axis=0, keepdims=True)
    col_best_for_player_2 = B == B.max(axis=1, keepdims=True)

    rows, cols = np.where(row_best_for_player_1 & col_best_for_player_2)
    return list(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))


def find_pareto(A: NDArray[np.integer], B: NDArray[np.integer]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    rows, cols = A.shape

    payoffs = [(i, j, A[i, j], B[i, j]) for i in range(rows) for j in range(cols)]

    for i, j, a_current, b_current in payoffs:
        dominated = any(
            (a_other >= a_current and b_other >= b_current)
            and (a_other > a_current or b_other > b_current)
            for _, _, a_other, b_other in payoffs
        )
        if not dominated:
            result.append((i, j))

    return result


def solve_mixed(A: NDArray[np.integer], B: NDArray[np.integer]):
    A_sym = Matrix(A.tolist()).applyfunc(Rational)
    B_sym = Matrix(B.tolist()).applyfunc(Rational)

    A_inv = A_sym.inv()
    B_inv = B_sym.inv()
    u = Matrix([1, 1])

    u_A_inv_u = (u.T * A_inv * u)[0]
    u_B_inv_u = (u.T * B_inv * u)[0]

    if u_A_inv_u == 0 or u_B_inv_u == 0:
        return None

    v1 = Rational(1) / u_A_inv_u
    v2 = Rational(1) / u_B_inv_u

    x = tuple(v2 * e for e in u.T * B_inv)
    y = tuple(v1 * e for e in A_inv * u)

    if not all(0 < p < 1 for p in (*x, *y)):
        return None

    if sum(x) != 1 or sum(y) != 1:
        return None

    x_vec = Matrix(list(x))
    y_vec = Matrix(list(y))
    expected_v1 = (x_vec.T * A_sym * y_vec)[0]
    expected_v2 = (x_vec.T * B_sym * y_vec)[0]

    return x, y, expected_v1, expected_v2


def print_game(game: Bimatrix) -> None:
    print(f"\n=== {game.name} ===")
    print("Payoff matrices (A/B):")
    for i in range(game.A.shape[0]):
        row = [f"({game.A[i, j]:>3},{game.B[i, j]:>3})" for j in range(game.A.shape[1])]
        print(" ".join(row))

def print_mixed(game: Bimatrix) -> None:
    result = solve_mixed(game.A, game.B)

    if not result:
        print("No interior mixed strategy equilibrium found.")
        return

    x, y, v1, v2 = result
    print("Mixed Strategy Nash Equilibrium:")
    print(f"  x = ({x[0]}, {x[1]}) = ({float(x[0]):.3f}, {float(x[1]):.3f})")
    print(f"  y = ({y[0]}, {y[1]}) = ({float(y[0]):.3f}, {float(y[1]):.3f})")
    print(f"  v1 = {v1} = {float(v1):.3f}")
    print(f"  v2 = {v2} = {float(v2):.3f}")


def random_game(seed: int = 42, size: int = 10, low: int = 0, high: int = 50) -> Bimatrix:
    rng = np.random.default_rng(seed)
    A = rng.integers(low, high + 1, size=(size, size), dtype=np.int64)
    B = rng.integers(low, high + 1, size=(size, size), dtype=np.int64)
    return Bimatrix(f"Random Game {size}x{size}", A, B)


def games() -> list[Bimatrix]:
    return [
        Bimatrix(
            "Battle of the Sexes",
            matrix([[4, 0], 
                    [0, 1]]),
            matrix([[1, 0], 
                    [0, 4]]),
        ),
        Bimatrix(
            "Traffic Game with Bias (ε=0.1)",
            np.array([[1, 0.9], 
                      [2, 0]]), 
            np.array([[1, 2], 
                      [0.9, 0]]),
        ),
        Bimatrix(
            "Prisoner's Dilemma",
            matrix([[-5, 0], 
                    [-10, -1]]),
            matrix([[-5, -10], 
                    [0, -1]]),
        ),
    ]

if __name__ == "__main__":
    for game in [random_game(seed=42), *games()]:
        print_game(game)
        nash = find_nash(game.A, game.B)
        pareto = find_pareto(game.A, game.B)
        print("Nash Equilibria:", one_based(nash))
        print("Pareto Optimal Outcomes:", one_based(pareto))
        print("Intersection of Sets:", one_based(sorted(set(nash) & set(pareto))))

    v = Bimatrix(
        "Variant 16",
        matrix([[8, 1], 
                [2, 3]]),
        matrix([[7, 2], 
                [0, 4]]),
    )
    print_game(v)
    nash = find_nash(v.A, v.B)
    pareto = find_pareto(v.A, v.B)
    print("Nash Equilibria:", one_based(nash))
    print("Pareto Optimal Outcomes:", one_based(pareto))
    print("Intersection of Sets:", one_based(sorted(set(nash) & set(pareto))))
    print_mixed(v)
