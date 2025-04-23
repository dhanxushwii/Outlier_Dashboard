import numpy as np
from scipy.ndimage import gaussian_filter

def reshape_scores(scores):
    """
    Reshape 1D outlier scores to 2D square grid for heatmap visualization.
    Pads with zeros if needed.
    """
    size = int(np.ceil(np.sqrt(len(scores))))
    padded = np.pad(scores, (0, size**2 - len(scores)), mode='constant', constant_values=0)
    return padded.reshape(size, size)

def apply_heatmap_style(scores, style):
    """
    Applies the selected heatmap style (raw, threshold, interpolated).
    """
    grid = reshape_scores(scores)

    if style == "raw":
        return grid

    elif style == "threshold":
        threshold = np.percentile(grid, 90)
        return np.where(grid >= threshold, grid, 0)

    elif style == "interpolated":
        return gaussian_filter(grid, sigma=1.5)

    else:
        raise ValueError(f"Unknown heatmap style: {style}")
