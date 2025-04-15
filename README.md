Supplementary Methods:

Although the simplified Python version of the code required us to complete only the branch rules in week 3, during actual debugging we discovered that the missing components in the model extended far beyond that. First, in the `make_segment` function, considering that the original source code stored vessels as one-dimensional arrays—which is counterintuitive and inconvenient for visualization and for determining neighboring relationships—I redefined a vessel as an independent coordinate array line segment while still maintaining the relationship between vessels and segments to facilitate subsequent calculations (S1). 

In the `solve_for_flow` function, blood flow in the upstream and downstream vessels is calculated using the Hagen–Poiseuille law. However, boundary conditions at the initial position and at vascular intersection points, which are essential for maintaining flow balance, were not taken into account; the original code still utilized one-dimensional array indexing to determine the upstream–downstream relationships. Here, by defining flow at the initial position using Pin and calculating the connections and flow distribution among vessels at specific positions, the correctness of blood flow is maintained.

EC migration needs to be determined based on flow. Endothelial cells first establish a basic migration vector based on the vessel they occupy and the flow vector; this vector is oriented opposite to the flow. In the source code, this determination was made solely according to the shape of the vessel. Considering that varying flow directions may exist within a vessel, I modified the vector to be directly opposite to the direction of flow Q. Subsequently, the migration angle is computed by taking into account the cell’s intrinsic inertia, random perturbations, and other factors. Through a matrix transformation, the migration angle influences the initial basic vector, and upon normalization, the final polarity vector is obtained. Given that the initial vector orientation of the cells is defined randomly, the polarity distribution in the first image is highly random, gradually converging under the influence of the basic vector and the angle computation.

The `cell_migration` module likely required the most substantial completion. It uses the polarity vector to decide whether a cell should migrate and to determine its migration target. Initially, it lacked directional judgment for EC migration under horizontal and rightward vertical conditions. It was then necessary to incorporate a diffusion balance (`down_seg`) to preserve a consistent concentration gradient. Furthermore, the judgment and updating of migration targets required refinement; for cells at specific positions (e.g., cells at positions 0 and 19), if they leave the boundary, a mechanism is needed for them to re-enter from the opposite side to maintain cell number stability. Additional logical checks and vector rotation transformations were also required for vessel segments at crossover points. Moreover, for cells migrating downstream in seg4 and those migrating upstream at position 15, additional branch rule judgment and migration target selection needed to be implemented.

At last, the `plot_network` function was optimized for vessel coordinates and segments to provide figure in an intuitive and easily interpretable format. The original source code was missing the visualization of distal vessels, which has now been supplemented. For the `abm_ec_simulation_v2` file, I adapted the subroutines in accordance with the modifications and added debugging functionalities to help identify potential bugs in the code construction process.
