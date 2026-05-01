from fractions import Fraction


def frac(x):
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def print_table(title, var_names, basis, c_basis, rows, c):
    m = len(rows)
    num_vars = len(var_names)
    F = [sum(c_basis[i] * rows[i][j] for i in range(m)) - c[j] for j in range(num_vars)]
    F_value = sum(c_basis[i] * rows[i][-1] for i in range(m))

    print("\n" + title)
    header = ["Базис"] + var_names + ["b"]
    sep = "-" * (13 * len(header))
    print(" | ".join(f"{h:>10}" for h in header))
    print(sep)
    for i in range(m):
        line = [var_names[basis[i]]] + [frac(rows[i][j]) for j in range(num_vars)] + [frac(rows[i][-1])]
        print(" | ".join(f"{x:>10}" for x in line))
    print(sep)
    diff_line = ["F"] + [frac(F[j]) for j in range(num_vars)] + [frac(F_value)]
    print(" | ".join(f"{x:>10}" for x in diff_line))
    return F, F_value


def nextIt(rows, change_row, change_col):
    value = rows[change_row][change_col]
    rows[change_row] = [v / value for v in rows[change_row]]
    for i in range(len(rows)):
        if i == change_row:
            continue
        coef = rows[i][change_col]
        rows[i] = [rows[i][j] - coef * rows[change_row][j] for j in range(len(rows[i]))]


def simplex(C):
    A = [[Fraction(x) for x in row] for row in C]
    m, n = len(A), len(A[0])
    b = [Fraction(1)] * m
    c = [Fraction(1)] * n + [Fraction(0)] * m

    var_names = [f"v{j+1}" for j in range(n+m)]
    num_vars = n + m
    rows = [A[i] + [Fraction(1) if i == k else Fraction(0) for k in range(m)] + [b[i]] for i in range(m)]
    basis = [n + i for i in range(m)]
    c_basis = [Fraction(0)] * m
    it = 0

    while True:
        F, value = print_table(f"Итерация {it}", var_names, basis, c_basis, rows, c)
        min_value = min(F)
        if min_value >= 0:
            break
        change_col = F.index(min_value)
        ratios = [(rows[i][-1] / rows[i][change_col], i) for i in range(m) if rows[i][change_col] > 0]
        _, change_row = min(ratios)
        nextIt(rows, change_row, change_col)
        basis[change_row] = change_col
        c_basis[change_row] = c[change_col]
        it += 1

    solution = [Fraction(0)] * num_vars
    for i in range(m):
        solution[basis[i]] = rows[i][-1]
    g = Fraction(1) / value
    y = [v / value for v in solution[:n]]
    return solution[:n], value, g, y


def dual_simplex(C):
    A = [[Fraction(-C[i][j]) for i in range(len(C))] for j in range(len(C[0]))]
    m, n = len(A), len(A[0])
    b = [Fraction(-1)] * len(C[0])
    c = [Fraction(-1)] * len(C) + [Fraction(0)] * m

    var_names = [f"u{j+1}" for j in range(n+m)]
    num_vars = n + m
    rows = [A[i] + [Fraction(1) if i == k else Fraction(0) for k in range(m)] + [b[i]] for i in range(m)]
    basis = [n + i for i in range(m)]
    c_basis = [Fraction(0)] * m
    it = 0

    while True:
        F, value = print_table(f"Итерация {it}", var_names, basis, c_basis, rows, c)
        min_b = min(row[-1] for row in rows)
        if min_b >= 0:
            break
        change_row = next(i for i in range(m) if rows[i][-1] == min_b)
        ratios = [(F[j] / (-rows[change_row][j]), j) for j in range(num_vars) if rows[change_row][j] < 0]
        _, change_col = min(ratios)
        nextIt(rows, change_row, change_col)
        basis[change_row] = change_col
        c_basis[change_row] = c[change_col]
        it += 1

    W = -value
    solution = [Fraction(0)] * num_vars
    for i in range(m):
        solution[basis[i]] = rows[i][-1]
    u = solution[:n]
    g = Fraction(1) / W
    x = [v / W for v in u]
    return u, W, g, x

if __name__ == "__main__":
    C = [
    [16, 17,  8, 15, 17],
    [ 0,  3, 19,  8,  2],
    [13, 19,  7, 15,  9],
    [11, 15,  2, 16,  2],
    ]

    C = [[Fraction(v) for v in row] for row in C]

    print("Решение задачи игрока A")
    u, W, g_a, x = dual_simplex(C)
    print("\nРезультат для игрока A:")
    print("u =", [frac(v) for v in u])
    print("g = 1 / W =", frac(g_a))
    print("x =", [frac(v) for v in x])

    print("Решение задачи игрока B")
    v, Z, g_b, y = simplex(C)
    print("\nРезультат для игрока B:")
    print("v =", [frac(v) for v in v])
    print("g = 1 / Z =", frac(g_b))
    print("y =", [frac(v) for v in y])

    print("Итоговое решение игры")
    print("x* =", [frac(v) for v in x])
    print("y* =", [frac(v) for v in y])
    print("g =", frac(g_a))

