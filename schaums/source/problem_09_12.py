"""
PROGRAM:    EHEAT
TITLE:      DEMO PROGRAM FOR EXPLICIT METHOD
            FOR HEAT EQUATION, UT = KAPPA*UXX
INPUT:      N NUMBER OF X-SUBINTERVALS
            K, TIME STEP
            TMAX, MAXIMUM COMPUTATION TIME
            KAPPA, DIFFUSIVITY VALUE
            (X1,X2), X-INTERVAL
            P(T), LEFT BOUNDARY CONDITION
            Q(T), RIGHT BOUNDARY CONDITION
            F(X), INITIAL CONDITION
            E(X,T), EXACT SOLUTION
OUTPUT:     NUMERICAL AND EXACT SOLUTION AT T=TMAX
"""
import math

# ----------------------------
# COMMON block equivalents
# ----------------------------
U = [0.0] * 52
V = [0.0] * 52

# ----------------------------
# DATA initialization
# ----------------------------
T     = 0.0
X1    = 0.0
X2    = 1.0
KAPPA = 1.0

PI = 4.0 * math.atan(1.0)

# ----------------------------
# Statement functions → lambdas
# ----------------------------
F = lambda x: 100.0 * math.sin(PI * x)
E = lambda x, t: 100.0 * math.exp(-PI * PI * t) * math.sin(PI * x)

P = lambda t: 0.0
Q = lambda t: 0.0

# ----------------------------
# Input
# ----------------------------
TMAX = float(input("Enter TMAX: "))
N    = int(input("Enter number of X-subintervals N: "))
K    = float(input("Enter time step K: "))

# ----------------------------
# Derived quantities
# ----------------------------
H = (X2 - X1) / N
R = KAPPA * K / (H * H)

# ----------------------------
# Initial condition
# DO 10 I = 0,N
# ----------------------------
for I in range(0, N + 1):
    X = X1 + I * H
    V[I] = F(X)

# ============================
# Time-stepping loop
# (Fortran labels 16–30)
# ============================
while True:

    # ------------------------
    # DO 20 I = 1,N-1
    # ------------------------
    for I in range(1, N):
        U[I] = V[I] + R * (V[I + 1] - 2.0 * V[I] + V[I - 1])

    # Time update and boundaries
    T = T + K
    U[0] = P(T)
    U[N] = Q(T)

    # ------------------------
    # DO 30 I = 0,N
    # ------------------------
    for I in range(0, N + 1):
        V[I] = U[I]

    # ------------------------
    # IF(ABS(TMAX-T).GT.K/2) GOTO 16
    # ------------------------
    if abs(TMAX - T) <= K / 2.0:
        break

# ============================
# Output section
# ============================
print("\n\n\n        RESULTS FROM PROGRAM EHEAT\n")
print(f"N={N:4d}          K = {K:8.6f}          TMAX ={TMAX:5.2f}\n")
print(f"T = {T:5.2f}        NUMERICAL            EXACT\n")

for I in range(0, N + 1):
    X = X1 + I * H
    EXACT = E(X, T)
    print(f"X = {X:4.1f}     {U[I]:13.6f}     {EXACT:13.6f}")


