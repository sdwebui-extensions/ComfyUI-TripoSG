import torch
import torch.nn as nn

def extract_near_surface_points(input_tensor, alpha):
    device = input_tensor.device
    
    # Add offset and handle invalid values
    val = input_tensor + alpha
    valid_mask = val > -9000  # Assume -9000 is invalid value
    
    # Get neighbors
    left = get_neighbor(val, 1, axis=0)
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)
    up = get_neighbor(val, -1, axis=2)
    
    # Handle boundary invalid values
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)
        
    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)
    
    # Calculate sign consistency
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)
    
    # Check if all signs are consistent
    same_sign = torch.all(neighbors_sign == sign, dim=0)
    
    # Generate final mask
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)

def get_neighbor(t, shift, axis):
    if shift == 0:
        return t.clone()
        
    pad_dims = [0, 0, 0, 0, 0, 0]  # Format: [x_before, x_after, y_before, y_after, z_before, z_after]
    
    if axis == 0:  # x-axis
        pad_idx = 0 if shift > 0 else 1
        pad_dims[pad_idx] = abs(shift)
    elif axis == 1:  # y-axis
        pad_idx = 2 if shift > 0 else 3
        pad_dims[pad_idx] = abs(shift)
    elif axis == 2:  # z-axis
        pad_idx = 4 if shift > 0 else 5
        pad_dims[pad_idx] = abs(shift)
        
    padded = nn.functional.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')
    
    slice_dims = [slice(None)] * 3
    if axis == 0:
        if shift > 0:
            slice_dims[0] = slice(shift, None)
        else:
            slice_dims[0] = slice(None, shift)
    elif axis == 1:
        if shift > 0:
            slice_dims[1] = slice(shift, None)
        else:
            slice_dims[1] = slice(None, shift)
    elif axis == 2:
        if shift > 0:
            slice_dims[2] = slice(shift, None)
        else:
            slice_dims[2] = slice(None, shift)
            
    padded = padded.squeeze(0).squeeze(0)
    sliced = padded[slice_dims]
    return sliced 