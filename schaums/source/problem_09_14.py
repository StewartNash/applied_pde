"""
PROGRAM:    ADI
TITLE:      DEMO PROGRAM FOR ADI METHOD FOR
            UT = KAPPA*(UXX * UYY)
INPUT:      MMAX & NMAX, NUMBER OF X & Y-SUBINTERVALS
            K, TIME STEP
            TMAX, MAXIMUM COMPUTATION TIME
            KAPPA, DIFFUSIVITY VALUE
            (X1,X2) & (Y1,Y2), X & Y-INTERVALS
            P1(Y,T) & Q1(Y,T), LEFT & RIGHT BOUNDARY CONDITIONS
            P2(X,T) & Q2(X,T), UPPER & LOWER BOUNDARY CONDITIONS
            F(X,Y), INITIAL CONDITION
            E(X,Y,T), EXACT SOLUTION
OUTPUT:     NUMERICAL AND EXACT SOLUTION AT T=TMAX
"""
import math

# ============================================================
# COMMON blocks
# ============================================================
# BLOCK1
A = [0.0] * 52
B = [0.0] * 52
C = [0.0] * 52
D = [0.0] * 52
L = 0

# BLOCK2
U = [[0.0 for _ in range(52)] for _ in range(52)]
V = [[0.0 for _ in range(52)] for _ in range(52)]

# ============================================================
# DATA initialization
# ============================================================
T     = 0.0
X1    = 0.0
X2    = 1.0
Y1    = 0.0
Y2    = 1.0
KAPPA = 1.0

PI = 4.0 * math.atan(1.0)

# ============================================================
# Statement functions → lambdas
# ============================================================
P1 = lambda y, t: 0.0
Q1 = lambda y, t: 0.0
P2 = lambda x, t: 0.0
Q2 = lambda x, t: 0.0

F = lambda x, y: 100.0 * math.sin(PI * x) * math.sin(PI * y)
E = lambda x, y, t: (
    100.0 * math.exp(-2.0 * PI * PI * t)
    * math.sin(PI * x)
    * math.sin(PI * y)
)

# ============================================================
# Input
# ============================================================
TMAX = float(input("ENTER TMAX: "))
K    = float(input("ENTER TIME STEP K: "))

MMAX = int(input("ENTER NUMBER OF X-SUBINTERVALS MMAX: "))
NMAX = int(input("ENTER NUMBER OF Y-SUBINTERVALS NMAX: "))

# ============================================================
# Grid spacing
# ============================================================
HX = (X2 - X1) / MMAX
HY = (Y2 - Y1) / NMAX

# ============================================================
# Initial condition
# ============================================================
for M in range(0, MMAX + 1):
    for N in range(0, NMAX + 1):
        X = X1 + M * HX
        Y = Y1 + N * HY
        U[M][N] = F(X, Y)

# ============================================================
# Tridiagonal solver (Thomas algorithm)
# ============================================================
def TRIDI():
    global A, B, C, D, L

    # Forward elimination
    for I in range(2, L + 1):
        RT = -A[I] / B[I - 1]
        B[I] = B[I] + RT * C[I - 1]
        D[I] = D[I] + RT * D[I - 1]

    # Back substitution
    D[L] = D[L] / B[L]
    for I in range(L - 1, 0, -1):
        D[I] = (D[I] - C[I] * D[I + 1]) / B[I]

# ============================================================
# Time-stepping loop (label 15)
# ============================================================
while True:

    # --------------------------------------------------------
    # Vertical sweep (solve in x-direction)
    # --------------------------------------------------------
    RX = KAPPA * K / (HX * HX)

    for N in range(1, NMAX):
        Y = Y1 + N * HY

        for M in range(1, MMAX):
            A[M] = -0.5 * RX
            B[M] = 1.0 + RX
            C[M] = -0.5 * RX
            D[M] = (
                0.5 * RX * (U[M - 1][N] + U[M + 1][N])
                + (1.0 - RX) * U[M][N]
            )

        L = MMAX - 1
        TRIDI()

        for M in range(1, MMAX):
            V[M][N] = D[M]

        V[0][N]     = P1(Y, T)
        V[MMAX][N]  = Q1(Y, T)

    # --------------------------------------------------------
    # Horizontal sweep (solve in y-direction)
    # --------------------------------------------------------
    RY = KAPPA * K / (HY * HY)

    for M in range(1, MMAX):
        X = X1 + M * HX

        for N in range(1, NMAX):
            A[N] = -0.5 * RY
            B[N] = 1.0 + RY
            C[N] = -0.5 * RY
            D[N] = (
                0.5 * RY * (V[M][N - 1] + V[M][N + 1])
                + (1.0 - RY) * V[M][N]
            )

        L = NMAX - 1
        TRIDI()

        for N in range(1, NMAX):
            U[M][N] = D[N]

        U[M][0]     = P2(X, T)
        U[M][NMAX]  = Q2(X, T)

    # --------------------------------------------------------
    # Advance time
    # --------------------------------------------------------
    T = T + K

    if abs(TMAX - T) <= K / 2.0:
        break

# ============================================================
# Output
# ============================================================
print("\n\n        RESULTS FROM PROGRAM ADI\n")
print(f"MMAX={MMAX:2d} NMAX={NMAX:2d}     K = {K:5.2f}     TMAX ={TMAX:5.2f}\n")
print(f"T = {T:5.2f}        NUMERICAL            EXACT\n")

for M in range(1, MMAX // 2 + 1):
    for N in range(1, M + 1):
        X = X1 + M * HX
        Y = Y1 + N * HY
        EXACT = E(X, Y, T)
        print(f"M,N = {M},{N}     {U[M][N]:13.6f}     {EXACT:13.6f}")

