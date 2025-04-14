import numpy as np

def make_segments(L):
    L = np.array(L)
    # Vessel 1: vertical from [0,0]
    y_v1 = np.cumsum([0] + list(L[0:5])) * 1e6
    vessel1 = np.array([[0, y] for y in y_v1])

    # Vessel 2: horizontal from vessel1[5]
    x_v2 = np.cumsum([0] + list(L[5:15])) * 1e6
    vessel2 = np.array([[vessel1[5,0] + x, 50] for x in x_v2])

    # Vessel 3: vertical from vessel2[10]
    y_v3 = 50 - np.cumsum([0] + list(L[15:20])) * 1e6
    vessel3 = np.array([[100, y] for y in y_v3])
    vessel3[0] = vessel2[10]


    # Vessel 4: vertical from vessel1[5]
    y_v4 = 50 + np.cumsum([0] + list(L[20:25])) * 1e6
    vessel4 = np.array([[0, y] for y in y_v4])
    vessel4[0] = vessel1[5]

    # Vessel 5: horizontal from vessel4[5]
    x_v5 = np.cumsum([0] + list(L[25:35])) * 1e6
    vessel5 = np.array([[vessel4[5,0] + x, 100] for x in x_v5])

    # Vessel 6: vertical from vessel5[10]
    y_v6 = 100 - np.cumsum([0] + list(L[35:40])) * 1e6
    vessel6 = np.array([[100, y] for y in y_v6])
    vessel6[0] = vessel5[10]

    return [vessel1, vessel2, vessel3, vessel4, vessel5, vessel6]