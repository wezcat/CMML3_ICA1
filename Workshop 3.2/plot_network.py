import numpy as np
import matplotlib.pyplot as plt

def plot_network(segments, D, P, Q, seg_cells, tau=None):
    """Mapping the vascular network, including pressure, flow, diameter, and cell polarity distribution"""
    
    plt.figure(figsize=(12, 6))
    
    # Left subgraph: pressure, flow, and diameter networks
    plt.subplot(1, 2, 1)
    plt.title('Pressure, Flow, Diameter of Network')
    
    # Unpack segments
    vessel1, vessel2, vessel3, vessel4, vessel5, vessel6 = segments
    
    # Vessel 1 (seg 0-4)
    for seg in range(5):
        if D[seg] != 0:
            color = "red" if Q[seg] > 0 else "blue"
            plt.plot([vessel1[seg, 0], vessel1[seg+1, 0]],
                     [vessel1[seg, 1], vessel1[seg+1, 1]],
                     color=color, linewidth=D[seg] * 1e6 / 2)
    
    # Vessel 2 (seg 5-14)
    for seg in range(5, 15):
        if D[seg] != 0:
            color = "red" if Q[seg] > 0 else "blue"
            plt.plot([vessel2[seg-5, 0], vessel2[seg-4, 0]],
                     [vessel2[seg-5, 1], vessel2[seg-4, 1]],
                     color=color, linewidth=D[seg] * 1e6 / 2)
    
    # Vessel 3 (seg 15-19)
    for seg in range(15, 20):
        if D[seg] != 0:
            color = "red" if Q[seg] > 0 else "blue"
            plt.plot([vessel3[seg-15, 0], vessel3[seg-14, 0]],
                     [vessel3[seg-15, 1], vessel3[seg-14, 1]],
                     color=color, linewidth=D[seg] * 1e6 / 2)
    
    # Vessel 4 (seg 20-24)
    for seg in range(20, 25):
        if D[seg] != 0:
            color = "red" if Q[seg] > 0 else "blue"
            plt.plot([vessel4[seg-20, 0], vessel4[seg-19, 0]],
                     [vessel4[seg-20, 1], vessel4[seg-19, 1]],
                     color=color, linewidth=D[seg] * 1e6 / 2)
    
    # Vessel 5 (seg 25-34)
    for seg in range(25, 35):
        if D[seg] != 0:
            color = "red" if Q[seg] > 0 else "blue"
            plt.plot([vessel5[seg-25, 0], vessel5[seg-24, 0]],
                     [vessel5[seg-25, 1], vessel5[seg-24, 1]],
                     color=color, linewidth=D[seg] * 1e6 / 2)
    
    # Vessel 6 (seg 35-39)
    for seg in range(35, 40):
        if D[seg] != 0:
            color = "red" if Q[seg] > 0 else "blue"
            plt.plot([vessel6[seg-35, 0], vessel6[seg-34, 0]],
                     [vessel6[seg-35, 1], vessel6[seg-34, 1]],
                     color=color, linewidth=D[seg] * 1e6 / 2)
    
    plt.grid(True)
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')
    plt.axis('equal')
    
    # Right subgraph: Cell polarity distribution
    plt.subplot(1, 2, 2)
    plt.title('Distribution of Cell Polarity')
    plt.axis([-1, 1, -1, 1])
    plt.grid(True)
    
    # Plot the polarity vector for each cell
    has_vectors = False
    for seg in range(len(seg_cells)):
        num_cells = seg_cells[seg]['num_cells']
        pv_shape = seg_cells[seg]['polarity_vectors'].shape
        if num_cells > 0:
            if pv_shape[1] != num_cells:
                print(f"Warning: Seg {seg} mismatch - num_cells: {num_cells}, polarity_vectors shape: {pv_shape}")
            for cell in range(min(num_cells, pv_shape[1])):
                polarity = seg_cells[seg]['polarity_vectors'][:, cell]
                if not np.all(polarity == 0):
                    plt.plot([0, polarity[0]], [0, polarity[1]], 'b-')
                    has_vectors = True
                else:
                    print(f"Zero vector skipped in seg {seg}, cell {cell}")
    if not has_vectors:
        print("No valid polarity vectors to plot!")

    plt.xlabel('u')
    plt.ylabel('v')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
