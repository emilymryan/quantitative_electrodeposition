import numpy as np
from skimage.measure import label, perimeter as sk_perimeter, regionprops
import pandas as pd
import matplotlib.pyplot as plt

"""""
We want to measure the growth density over a large region of interest (ROI), 
both the ratio of white pixels to row length, as well as the frequency of big dendrites.
We also want the thickness, number,
and angle compared to the main trunk of the dendrite for each individual dendrite.
Another thing is the overall aspect ratio (x/y) of the full dendrite.
"""""
def dendrite_height_width_ratios(clusters, um_per_px=None, image_name=""):
    """
    clusters: list[2D bool/0-1 array], each a connected component mask
    um_per_px: float or None. If provided, adds *_um columns.
    image_name: optional image id for output table.
    """
   
    rows = []
    for i, m in enumerate(clusters):
        if m is None or m.size == 0 or not np.any(m):
            continue
        ys, xs = np.where(m)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        height_px = (ymax - ymin + 1)
        width_px  = (xmax - xmin + 1)
        ratio     = height_px / width_px if width_px > 0 else np.nan

        rec = {
            "image": image_name,
            "cluster_id": i,
            "height_px": int(height_px),
            "width_px":  int(width_px),
            "height_to_width_ratio": float(ratio),
        }
        if um_per_px is not None and np.isfinite(um_per_px):
            rec["height_um"] = height_px * um_per_px
            rec["width_um"]  = width_px  * um_per_px
            # ratio is unitless; same value in µm/µm
        rows.append(rec)

    return pd.DataFrame(rows)

def row_white_ratio(simulation: bool,mask: np.ndarray,
                    scale_um_per_px: float,
                    bin_um: float = 0.01):
    mask = mask.astype(bool)
    H, W = mask.shape
    # per-row white ratio
    row_counts = mask.sum(axis=1)       # white pixels per row
    per_row_ratio = row_counts / float(W)
    # --- keep only rows BEFORE the first all-white row ---
    if not simulation:
        upper_contains_white_idx = np.where(per_row_ratio > 0.1)[0][0]
        black_idices = np.where(per_row_ratio <= 0.05)[0]
        for black_ind in black_idices:
            if black_ind > upper_contains_white_idx:
                stop_idx = black_ind
                break
        per_row_ratio = per_row_ratio[:stop_idx]
        H = stop_idx # update height to what we keep

    # physical position (use row CENTER) in µm
    y_um = (H - np.arange(H) - 0.5) * scale_um_per_px
    total_um = H * scale_um_per_px

    # fixed physical-height bins
    bins = np.arange(0.0, total_um + bin_um, bin_um)
    bin_idx = np.digitize(y_um, bins) - 1  # bin for each row

    # aggregate by bin (skip empty bins)
    centers, ratios, counts = [], [], []
    for b in range(max(bin_idx.min(), 0), max(bin_idx.max(), -1) + 1):
        sel = (bin_idx == b)
        if not np.any(sel):
            continue
        centers.append(bins[b] + bin_um / 2.0)
        ratios.append(per_row_ratio[sel].mean())
        counts.append(int(sel.sum()))

    # keep all the ones before the first thats all white
    
    overall_mean = float(per_row_ratio.mean())
    return (np.asarray(centers),
            np.asarray(ratios),
            overall_mean,
            np.asarray(counts))


def frequency_of_big_dendrites(scale, num_dendrites):
    if not scale:
        return float("nan")     # or 0.0 if you prefer
    mic = scale.get("microns")
    if mic in (None, 0):
        return float("nan")
    return num_dendrites / float(mic)

def skeleton_aspect_ratio_bbox(skeleton: np.ndarray) -> float:
    skel = np.asarray(skeleton, dtype=bool)
    if not skel.any():
        return np.nan

    ys, xs = np.nonzero(skel)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)

    return float(w / h) if h > 0 else np.nan

