import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import random
from ex4_utils import kalman_step


def kalman_init(model_name, q, r):
    if model_name == "RW":
        # [x, y]
        F = sp.zeros(2,2)
        L = sp.eye(2)
        C = np.eye(2)

    elif model_name == "NCV":
        # [x, y, x', y']
        F = sp.zeros(4,4)
        F[0,2] = F[1,3] = 1
        L = sp.Matrix([[0,0,1,0],[0,0,0,1]]).T
        C = np.array([[1,0,0,0],[0,1,0,0]])

    elif model_name == "NCA":
        #  [x, y, x', y', x'', y'']
        F = sp.zeros(6,6)
        F[0,2] = F[1,3] = F[2,4] = F[3,5] = 1
        L = sp.Matrix([[0,0,0,0,1,0],[0,0,0,0,0,1]]).T
        C = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
    else:
        raise Exception("Model name not recognized")
    
    T, q_sym = sp.symbols("T q")
    Fi = sp.exp(F*T)
    Q = sp.integrate((Fi*L)*q_sym*(Fi*L).T, (T, 0, T))
    A = Fi.evalf(subs={T:1,q_sym:q})
    A = np.array(A.tolist()).astype(np.float64)
    Q = Q.evalf(subs={T:1,q_sym:q})
    Q = np.array(Q.tolist()).astype(np.float64)
    R = r * np.eye(2)

    return A, C, Q, R

def kalman_eval(x, y, model, q, r):
    A, C, Q_i, R_i = kalman_init(model, q, r)
    sx = np.zeros((x.size, 1), dtype=np.float32 ).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32 ).flatten()
    sx[0] = x[0]
    sy[0] = y[0]
    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()
    state[0] = x[0]
    state[1] = y[0]
    covariance = np.eye(A.shape[0], dtype=np.float32)
    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i,
            np.reshape(np.array([x[j], y[j]]), (-1, 1)),
            np.reshape(state, (-1, 1)), covariance)
        sx[j]= state[0]
        sy[j]= state[1]

    return sx, sy

# Original path
N = 40
v = np.linspace(5*math.pi, 0 ,N)
x = np.cos(v)*v
y = np.sin(v)*v

# Test on a rectangle
# x = np.array([0,0,100,100,0])
# y = np.array([0,100,100,0,0])

# Random path
# N = 20
# random.seed(42)
# x = np.sort(random.sample(range(0,100), N))
# y = np.array(random.sample(range(0,100), N))


models = ["RW", "NCV", "NCA"]
qs = [100, 5,  1, 1, 1]
rs = [1, 1, 1, 5, 100]


fig, axs = plt.subplots(len(models),len(qs))
for i in range(len(models)):
    curr_model = models[i]
    for j in range(len(qs)):
        curr_q = qs[j]
        curr_r = rs[j]
        sx, sy = kalman_eval(x, y, curr_model, curr_q, curr_r)
        axs[i,j].plot(x, y, color="blue", linestyle="-", marker="o", mfc="none")
        axs[i,j].plot(sx, sy, color="red", linestyle="-", marker="o", mfc="none")
        axs[i,j].title.set_text(f"{curr_model}: q={curr_q}, r={curr_r}")
plt.show()
