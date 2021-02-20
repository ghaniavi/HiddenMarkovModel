import pandas as pd
import numpy as np
 

def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
 
    return alpha
 

def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
 
    return beta
 

def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)
 
    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        b = np.divide(b, denominator.reshape((-1, 1)))
 
    return (a, b)
 

def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
 
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        #print(s)
        if s == 0:
            result.append("Aktivitas Memasak: Lampu Menyala")
        elif s == 1:
            result.append("Aktivitas Makan: Lampu Menyala")
        elif s == 2:
            result.append("Aktivitas Istirahat: Lampu Mati")
        elif s == 3:
            result.append("Aktivitas Cuci Piring: Lampu Menyala")
        elif s == 4:
            result.append("Aktivitas Menonton TV: Lampu Menyala")
        else:
            result.append("Null")
    return result
 

data = pd.read_csv('dataAktivitas.csv')

V = data['Zona'].values
 
# Transition Probabilities
aa = np.asmatrix([[0.5, 0.0625, 0.09375, 0.15625, 0.1875],
                 [0.083333, 0.5, 0.083333, 0.083333, 0.25],
                 [0.136364, 0, 0.5, 0.272727, 0.090909],
                 [0.266667, 0.033333, 0.133333, 0.5, 0.066667],
                 [0.153846, 0.076923, 0.115385, 0.153846, 0.5]])
a = np.squeeze(np.asarray(aa))
#a = np.ones((2, 2))
print("Probabilitas Aktivitas ke Aktivitas")
print(a)
print("")
 
# Emission Probabilities
bb = np.asmatrix([[0.394737, 0.394737, 0.210526],
                [0.111111, 0.555556, 0.333333],
                [0.692308, 0.153846, 0.153846],
                [0.71875, 0.09375, 0.1875],
                [0.205128, 0.384615,  0.410256]])
b = np.squeeze(np.asarray(bb))
#b = np.array(((1, 3, 5), (2, 4, 6)))
print("Probabilitas Aktivitas ke Zona")
print(b)
print("")
 
# Equal Probabilities for the initial distribution
initial_distribution = np.array((0.262295, 0.098361, 0.180328, 0.245902, 0.213115))
print("Probabilitas Awal")
print(initial_distribution)
 
a, b = baum_welch(V, a, b, initial_distribution, n_iter=100)
print("")
print(viterbi(V, a, b, initial_distribution))