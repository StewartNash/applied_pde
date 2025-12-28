"""
PROGRAM:    IHEAT
TITLE:      DEMO PROGRAM FOR IMPLICIT AND CRANK-
            NICOLSON METHODS FOR UT = KAPPA*UXX
INPUT:      N, NUMBER OF X-SUBINTERVALS
            K, TIME STEP
            TMAX, MAXIMUM COMPUTATION TIME
            KAPPA, DIFFUSIVITY VALUE
            (X1, X2), X-INTERVAL
            P(T), LEFT BOUNDARY CONDITION
            Q(T), RIGHT BOUNDARY CONDITION
            F(X), INITIAL CONDITION
            E(X,T), EXACT SOLUTION
            W, W=1 FOR IMPLICIT-W=.5 FOR CRANK-NICOLSON
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
U = [0.0] * 52

# ============================================================
# DATA initialization
# ============================================================
T     = 0.0
X1    = 0.0
X2    = 1.0
KAPPA = 1.0

# Constants
PI = 4.0 * math.atan(1.0)

# ============================================================
# Statement functions → lambdas
# ============================================================
P = lambda t: 0.0
Q = lambda t: 0.0
F = lambda x: 100.0 * math.sin(PI * x)
E = lambda x, t: 100.0 * math.exp(-PI * PI * t) * math.sin(PI * x)

# ============================================================
# Input
# ============================================================
TMAX = float(input("ENTER TMAX: "))
N    = int(input("ENTER NUMBER OF X-SUBINTERVALS: "))
K    = float(input("ENTER THE TIME STEP K: "))

print("ENTER 1 FOR IMPLICIT, .5 FOR CRANK-NICOLSON METHOD")
W = float(input())

# ============================================================
# Derived quantities
# ============================================================
H = (X2 - X1) / N
R = KAPPA * K / (H * H)

# ============================================================
# Set initial condition
# DO 10 I = 0,N
# ============================================================
for I in range(0, N + 1):
    X = X1 + I * H
    U[I] = F(X)

# ============================================================
# Time-stepping loop (label 15)
# ============================================================
while True:

    # --------------------------------------------------------
    # Define tridiagonal linear system
    # --------------------------------------------------------
    L = N - 1

    for I in range(1, L + 1):
        A[I] = -W * R
        B[I] = 1.0 + 2.0 * W * R
        C[I] = -W * R
        D[I] = (
            U[I]
            + (1.0 - W) * R * (U[I - 1] - 2.0 * U[I] + U[I + 1])
        )

    # --------------------------------------------------------
    # CALL TRIDI
    # --------------------------------------------------------
    def TRIDI():
        global A, B, C, D, L

        # Forward substitution
        for I in range(2, L + 1):
            RT = -A[I] / B[I - 1]
            B[I] = B[I] + RT * C[I - 1]
            D[I] = D[I] + RT * D[I - 1]

        # Back substitution
        D[L] = D[L] / B[L]
        for I in range(L - 1, 0, -1):
            D[I] = (D[I] - C[I] * D[I + 1]) / B[I]

    TRIDI()

    # --------------------------------------------------------
    # Write solution at time T+K into U-array
    # --------------------------------------------------------
    T = T + K

    for I in range(1, N):
        U[I] = D[I]

    U[0] = P(T)
    U[N] = Q(T)

    # --------------------------------------------------------
    # Time-step termination test
    # IF(ABS(TMAX-T).GT.K/2) GOTO 15
    # --------------------------------------------------------
    if abs(TMAX - T) <= K / 2.0:
        break

# ============================================================
# Output section
# ============================================================
print("\n\n    RESULTS FROM PROGRAM IHEAT W={:5.2f}\n".format(W))
print(f"N = {N:4d}          K = {K:8.6f}          TMAX ={TMAX:5.2f}\n")
print(f"T = {T:5.2f}        NUMERICAL            EXACT\n")

for I in range(0, N + 1):
    X = X1 + I * H
    EXACT = E(X, T)
    print(f"X = {X:4.1f}     {U[I]:13.6f}     {EXACT:13.6f}")

