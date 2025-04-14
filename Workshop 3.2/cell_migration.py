import numpy as np
def cell_migration(seg, seg_cells, new_seg_cells, migrate, Q, tau, branch_rule, alpha):
    cell_size = 10e-6  # 迁移距离阈值
    mchance = 1  # 迁移概率

    num_cells = seg_cells[seg]['num_cells']
    if num_cells == 0:
        new_seg_cells[seg]['polarity_vectors'] = np.zeros((2, 0))
        new_seg_cells[seg]['num_cells'] = 0
        new_seg_cells[seg]['migration_indicators'] = np.zeros(0, dtype=int)
        return seg_cells, new_seg_cells

    # 初始化迁移指示器
    new_seg_cells[seg]['migration_indicators'] = np.zeros(num_cells, dtype=int)

    # 计算迁移意愿（与原代码一致）
    for cell in range(num_cells):
        mcell = np.random.rand()
        if mcell <= mchance:
            polar_vect = seg_cells[seg]['polarity_vectors'][:, cell]
            migrate_vect = cell_size * polar_vect

            # 针对不同 vessel 类型
            if (seg <= 4) or (20 <= seg <= 24):
                if migrate_vect[1] >= cell_size / 2:
                    new_seg_cells[seg]['migration_indicators'][cell] = 1
                    migrate[seg] += 1
                elif migrate_vect[1] <= -cell_size / 2:
                    new_seg_cells[seg]['migration_indicators'][cell] = -1
                    migrate[seg] += 1

            if (5 <= seg <= 14) or (25 <= seg <= 34):
                if migrate_vect[0] >= cell_size / 2:
                    new_seg_cells[seg]['migration_indicators'][cell] = 1
                    migrate[seg] += 1
                elif migrate_vect[0] <= -cell_size / 2:
                    new_seg_cells[seg]['migration_indicators'][cell] = -1
                    migrate[seg] += 1

            if (15 <= seg <= 19) or (seg >= 35):
                if migrate_vect[1] >= cell_size / 2:
                    new_seg_cells[seg]['migration_indicators'][cell] = -1
                    migrate[seg] += 1
                elif migrate_vect[1] <= -cell_size / 2:
                    new_seg_cells[seg]['migration_indicators'][cell] = 1
                    migrate[seg] += 1

    # 记录迁移细胞的信息
    migrating_cells = []
    for cell in range(num_cells):
        if new_seg_cells[seg]['migration_indicators'][cell] != 0:
            migrating_cells.append((cell, new_seg_cells[seg]['migration_indicators'][cell]))

    # Diffusion scheme处理
    down_seg = None
    if seg not in [19, 39]:
        down_seg = seg + 1
    elif seg == 19:
        down_seg = 0
    elif seg == 39:
        down_seg = 14 if (seg_cells[14]['num_cells'] < seg_cells[15]['num_cells'] and seg_cells[14]['num_cells'] > 0) else 15
    if seg == 4:
        down_seg = 5 if (seg_cells[5]['num_cells'] < seg_cells[20]['num_cells'] and seg_cells[5]['num_cells'] > 0) else 20

    if down_seg is not None and num_cells > 0:
        if seg_cells[down_seg]['num_cells'] < seg_cells[seg]['num_cells'] and seg_cells[down_seg]['num_cells'] > 0:
            idx = np.random.randint(0, num_cells)
            new_seg_cells[seg]['migration_indicators'][idx] = 0
            migrate[seg] -= 1

    # 保存原有num_cells用于后续未迁移细胞处理
    original_num_cells = num_cells

    # 分离出未迁移的细胞（indicator == 0）
    remaining_vectors = []
    for cell in range(original_num_cells):
        if new_seg_cells[seg]['migration_indicators'][cell] == 0:
            remaining_vectors.append(seg_cells[seg]['polarity_vectors'][:, cell])

    # 更新当前段信息：仅保留未迁移细胞
    if remaining_vectors:
        new_seg_cells[seg]['polarity_vectors'] = np.column_stack(remaining_vectors)
        new_seg_cells[seg]['num_cells'] = len(remaining_vectors)
        new_seg_cells[seg]['migration_indicators'] = np.zeros(len(remaining_vectors), dtype=int)
    else:
        new_seg_cells[seg]['polarity_vectors'] = np.zeros((2, 0))
        new_seg_cells[seg]['num_cells'] = 0
        new_seg_cells[seg]['migration_indicators'] = np.zeros(0, dtype=int)

    # 处理迁移的细胞（使用之前的 migrating_cells 列表）
    for cell, direction in migrating_cells:
        cell_vect = seg_cells[seg]['polarity_vectors'][:, cell]
        target = None
        if direction == 1:  # 向下游迁移
            target = seg + 1 if seg != 19 else 0
            if seg == 19:
                target = 0
            elif seg == 4:
                if branch_rule == 2:
                    target = 20 if np.dot(cell_vect, [0, 1]) > np.dot(cell_vect, [1, 0]) else 5
                elif branch_rule == 4:
                    target = 20 if np.random.rand() < 0.3 else 5
            elif seg == 39:
                if branch_rule == 2:
                    target = 15 if np.dot(cell_vect, [0, 1]) > np.dot(cell_vect, [1, 0]) else 14
                elif branch_rule == 4:
                    target = 15 if np.random.rand() < 0.3 else 14
        elif direction == -1:  # 向上游迁移
            target = seg - 1 if seg != 0 else 19
            if seg == 20:
                target = 4
            elif seg == 15:
                if branch_rule == 2:
                    target = 39 if np.dot(cell_vect, [0, 1]) > np.dot(cell_vect, [-1, 0]) else 14
                elif branch_rule == 4:
                    target = 39 if np.random.rand() < 0.3 else 14
            elif seg == 0:
                target = 19

        if target is not None:
            # 根据分支调整细胞方向
            theta = 0
            if direction == 1:
                if (seg == 4 and target == 5) or (seg == 24 and target == 25) or \
                   (seg == 14 and target in [15, 39]) or (seg == 34 and target == 35):
                    theta = -np.pi / 2
                elif seg == 19 and target == 0:
                    theta = -np.pi
            elif direction == -1:
                if (seg in [15, 39] and target == 14) or (seg == 35 and target == 34) or \
                   (seg == 25 and target == 24) or (seg == 5 and target in [4, 20]):
                    theta = np.pi / 2
                elif seg == 0 and target == 19:
                    theta = np.pi

            # 如果需要旋转，更新细胞的极性向量
            if theta != 0:
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                cell_vect = rot @ cell_vect
                norm = np.linalg.norm(cell_vect)
                if norm > 0:
                    cell_vect /= norm
                else:
                    cell_vect = np.array([1, 0])
                    print(f"Warning: Zero vector detected in seg {seg}, cell {cell}")

            # 将迁移的细胞加入目标段（target）
            if new_seg_cells[target]['num_cells'] == 0:
                new_seg_cells[target]['polarity_vectors'] = cell_vect[:, None]
                new_seg_cells[target]['migration_indicators'] = np.zeros(1, dtype=int)
            else:
                new_seg_cells[target]['polarity_vectors'] = np.column_stack(
                    (new_seg_cells[target]['polarity_vectors'], cell_vect))
                new_seg_cells[target]['migration_indicators'] = np.zeros(
                    new_seg_cells[target]['num_cells'] + 1, dtype=int)
            new_seg_cells[target]['num_cells'] += 1

    # 最后返回更新后的数据结构
    return seg_cells, new_seg_cells

