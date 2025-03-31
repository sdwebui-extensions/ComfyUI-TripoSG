import torch
import numpy as np
import torch.nn as nn

def process_grid_points(xyz_samples, device, dtype, batch_size, mini_grid_num):
    """
    Process and reshape grid points for FlashVDM processing
    """
    xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
    mini_grid_size = xyz_samples.shape[0] // mini_grid_num
    xyz_samples = xyz_samples.view(
        mini_grid_num, mini_grid_size,
        mini_grid_num, mini_grid_size,
        mini_grid_num, mini_grid_size, 3
    ).permute(
        0, 2, 4, 1, 3, 5, 6
    ).reshape(
        -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
    )
    return xyz_samples, mini_grid_size

def reshape_grid_logits(batch_logits, batch_size, grid_size, mini_grid_num, mini_grid_size):
    """
    Reshape the batch logits into a grid
    """
    grid_logits = torch.cat(batch_logits, dim=0).reshape(
        mini_grid_num, mini_grid_num, mini_grid_num,
        mini_grid_size, mini_grid_size,
        mini_grid_size
    ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
        (batch_size, grid_size[0], grid_size[1], grid_size[2])
    )
    return grid_logits

def group_points_for_processing(next_points, query_grid_num):
    """
    Group points for efficient processing with FlashVDM
    """
    min_val = next_points.min(axis=0).values
    max_val = next_points.max(axis=0).values
    vol_queries_index = (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
    index = torch.floor(vol_queries_index).long()
    index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
    index = index.sort()
    next_points = next_points[index.indices].unsqueeze(0).contiguous()
    unique_values = torch.unique(index.values, return_counts=True)
    return next_points, index, unique_values 