import cv2
import numpy as np
from scipy.spatial import KDTree
import matplotlib
import matplotlib.pyplot as plt


def interp_knn(input: np.ndarray, mask: np.ndarray, k=3, max_dist=None, ignore_mask=None):
    # max_dist: maximum distance to consider for interpolation
    # ignore_mask: mask to ignore for interpolation

    assert k > 1, "k should be greater than 1"
    valid_mask = mask > 0
    invalid_mask = mask <= 0    
    ignore_mask = np.zeros_like(mask) if ignore_mask is None else ignore_mask
    invalid_mask = invalid_mask & ~ignore_mask
    
    # interpolation
    val_x, val_y = np.where(valid_mask)
    inval_x, inval_y = np.where(invalid_mask)
    val_pos = np.stack([val_x, val_y], axis=1)
    inval_pos = np.stack([inval_x, inval_y], axis=1)

    tree = KDTree(val_pos)
    dists, inds = tree.query(inval_pos, k=k) # (N, k), (N, k)

    dists = np.where(dists == 0, 1e-10, dists)
    dists_max = np.max(dists, axis=-1)
    
    if max_dist is not None:
        dists_valid = dists_max <= max_dist    
        dists = dists[dists_valid]
        inds = inds[dists_valid]
        inval_x = inval_x[dists_valid]
        inval_y = inval_y[dists_valid]
    
    weights = 1 / dists
    weights /= np.sum(weights, axis=1, keepdims=True)
    nearest_vals = input[val_x[inds], val_y[inds]]
    weighted_avg = np.sum(nearest_vals * weights[..., None], axis=1)

    output_val = input.copy()
    output_val[inval_x, inval_y] = weighted_avg
    output_mask = mask.copy()
    output_mask[inval_x, inval_y] = 1
    
    return output_val, output_mask



def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """    
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]


def save_numpy_image(x, path='vis.png'):
    x = (x * 255).astype(np.uint8)
    if x.ndim == 2:
        cv2.imwrite(path, x)
    else:
        cv2.imwrite(path, x[..., [2, 1, 0]])
