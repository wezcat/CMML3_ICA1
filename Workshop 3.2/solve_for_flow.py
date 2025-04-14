import numpy as np
from scipy.linalg import solve

def solve_for_flow(G, Pin, Pout, H):
    Nn = 40
    Nseg = 40
    G = np.where(G == 0, 1e-25, G)  # 避免 G 为零导致奇异矩阵

    C = np.zeros((Nn, Nn))  # 系数矩阵
    B = np.zeros(Nn)  # 右侧向量

    # Node 0
    C[0,0] = G[0]
    B[0] = G[0] * Pin

    # Node 1
    C[1,1] = G[0] + G[1]
    C[1,2] = -G[1]
    B[1] = G[0] * Pin

    # Nodes 2 to 4
    for seg in range(2, 5):
        C[seg, seg-1] = -G[seg-1]
        C[seg, seg] = G[seg-1] + G[seg]
        C[seg, seg+1] = -G[seg]

    # Node 5 (junction)
    C[5,4] = -G[4]
    C[5,5] = G[4] + G[5] + G[20]
    C[5,6] = -G[5]
    C[5,21] = -G[20]

    # Nodes 6 to 14
    for seg in range(6, 15):
        C[seg, seg-1] = -G[seg-1]
        C[seg, seg] = G[seg-1] + G[seg]
        C[seg, seg+1] = -G[seg]

    # Node 15 (junction)
    C[15,14] = -G[14]
    C[15,15] = G[14] + G[15] + G[39]
    C[15,16] = -G[15]
    C[15,39] = -G[39]

    # Nodes 16 to 18
    for seg in range(16, 19):
        C[seg, seg-1] = -G[seg-1]
        C[seg, seg] = G[seg-1] + G[seg]
        C[seg, seg+1] = -G[seg]

    # Node 19
    C[19,18] = -G[18]
    C[19,19] = G[18] + G[19]
    B[19] = G[19] * Pout

    # Node 20
    C[20,20] = G[19]
    B[20] = G[19] * Pout

    # Node 21
    C[21,5] = -G[20]
    C[21,21] = G[20] + G[21]
    C[21,22] = -G[21]

    # Nodes 22 to 38
    for seg in range(22, 39):
        C[seg, seg-1] = -G[seg-1]
        C[seg, seg] = G[seg-1] + G[seg]
        C[seg, seg+1] = -G[seg]

    # Node 39 (junction)
    C[39,15] = -G[39]
    C[39,38] = -G[38]
    C[39,39] = G[38] + G[39]

    P = solve(C, B)  # 求解压力

    Q = np.zeros(Nseg)  # 计算流量
    for seg in range(20):
        Q[seg] = -G[seg] * (P[seg+1] - P[seg])
    Q[20] = -G[20] * (P[21] - P[5])
    for seg in range(21, 39):
        Q[seg] = -G[seg] * (P[seg+1] - P[seg])
    Q[39] = -G[39] * (P[15] - P[39])

    tau = H * Q  # 计算剪切应力
    return P, Q, tau