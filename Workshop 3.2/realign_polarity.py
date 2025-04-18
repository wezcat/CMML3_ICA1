import numpy as np

def realign_polarity(seg, Q, seg_cells, new_seg_cells, w1, w2, w3, w4):
    num_cells = seg_cells[seg]['num_cells']
    
    #Check if there are cells and the data shape is correct
    if num_cells > 0 and seg_cells[seg]['polarity_vectors'].shape[1] == num_cells:
        # Example Initialize the traffic direction
        if (0 <= seg <= 4) or (20 <= seg <= 24):
            flow_vect = -np.sign(Q[seg]) * np.array([0, 1])
        elif (5 <= seg <= 14) or (25 <= seg <= 34):
            flow_vect = -np.sign(Q[seg]) * np.array([1, 0])
        elif (15 <= seg <= 19) or (35 <= seg <= 39):
            flow_vect = -np.sign(Q[seg]) * np.array([0, -1])
        else:
            flow_vect = np.array([0, 0])
        
        # Normalized traffic direction
        flow_norm = np.linalg.norm(flow_vect)
        if flow_norm > 0:
            flow_vect = flow_vect / flow_norm
        
        # Go through every cell
        for cell in range(num_cells):
            polar_vect = seg_cells[seg]['polarity_vectors'][:, cell].copy()
            
            # Calculate the Angle phi2
            polar_norm = np.linalg.norm(polar_vect)
            if polar_norm > 0:
                polar_vect = polar_vect / polar_norm
            else:
                polar_vect = flow_vect.copy()
            
            dot_product = np.dot(polar_vect, flow_vect)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            phi2 = np.arccos(dot_product)
            
            # Calculate random Angle phi4
            phi4 = np.random.uniform(-np.pi, np.pi)
            
            # Calculate the rotation Angle theta
            theta = w1 * 0 + w2 * phi2 + w3 * 0 + w4 * phi4
            
            # Debugging information
            print(f"Seg {seg}, Cell {cell}: phi2={phi2:.4f}, phi4={phi4:.4f}, theta={theta:.4f}")
            
            # Transformation matrix
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
            
            # Renewal polarity vector
            new_polar_vect = rot @ polar_vect
            
            # Ensure normalization
            norm = np.linalg.norm(new_polar_vect)
            if norm > 0:
                new_polar_vect = new_polar_vect / norm
            else:
                new_polar_vect = flow_vect.copy()
            
            # Save to new_seg_cells
            new_seg_cells[seg]['polarity_vectors'][:, cell] = new_polar_vect
    
    return seg_cells, new_seg_cells
