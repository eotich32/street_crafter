import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    mask: np.array
    
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    mask = vertices['mask'].astype(np.bool_)
    return BasicPointCloud(points=positions, colors=colors, mask=mask)

def storePly(path, xyz, rgb, mask):
    # set rgb to 0 - 255
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)

    # set mask to bool data type    
    mask = mask.astype(np.bool_)
        
    # Define the dtype for the structured array
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('mask', '?')
    ]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb, mask), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)