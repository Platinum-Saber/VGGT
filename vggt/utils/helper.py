"""Small NumPy helpers used by the demos."""

import numpy as np


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """Keep at most ``max_trues`` randomly chosen True entries of a boolean mask."""
    true_indices = np.flatnonzero(mask)
    if true_indices.size <= max_trues:
        return mask
    keep = np.random.choice(true_indices, size=max_trues, replace=False)
    limited = np.zeros(mask.size, dtype=bool)
    limited[keep] = True
    return limited.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    """(num_frames, height, width, 3) array of (x, y, frame index) for every pixel."""
    y, x = np.indices((height, width), dtype=np.float32)
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    f = np.broadcast_to(np.arange(num_frames, dtype=np.float32)[:, None, None], (num_frames, height, width))
    return np.stack((x, y, f), axis=-1)
