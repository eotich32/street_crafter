import numpy as np
def project_numpy(xyz, K, RT, H, W):
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)
    
    return xyz_pixel, mask
def transform_points(points, transform_matrix):
    import torch

    """
    Apply a 4x4 transformation matrix to 3D points.

    Args:
        points: (N, 3) tensor of 3D points
        transform_matrix: (4, 4) transformation matrix

    Returns:
        (N, 3) tensor of transformed 3D points
    """
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homo_points = torch.cat([points, ones], dim=1)  # N x 4
    transformed_points = torch.matmul(homo_points, transform_matrix.T)
    return transformed_points[:, :3]

def transform_points_numpy(points, transform_matrix):
    homo_points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    transformed_points = homo_points @ transform_matrix.T
    return transformed_points[:, :3]
