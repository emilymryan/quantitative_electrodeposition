import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.ndimage import convolve, distance_transform_edt
from skimage.io import imsave
from skimage.measure import label, perimeter as sk_perimeter, regionprops
from skimage.morphology import binary_dilation, disk, medial_axis, remove_small_objects, skeletonize
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
import time
from collections import deque
from matplotlib.colors import ListedColormap
import random
from branch_tree import Branch
from typing import Tuple, List, Dict
from matplotlib.patches import Arc
from scipy.sparse import lil_matrix
from skimage.morphology import opening, closing, disk
from branches_core import (
num_branches,
split_skeleton_components,
find_root_for_graph,
create_branch_tree,
DFS_branch_tree,
compute_brach_atributes,
designate_branches_map_to_coord,
assign_binary_to_branches,
reshape_coords,
remove_if_small_branch,
remove_if_small_branch_extra
)

# -------------------------
# Main pipeline function
# -------------------------

def branches_info(cluster, output_path,large_file, show, save):
    
    binary_cluster = cluster > 0

    if large_file:
        binary_smooth = opening(binary_cluster, disk(1))
        binary_smooth = closing(binary_smooth, disk(2))
        binary_cluster = binary_smooth
        
    skeleton = skeletonize(binary_cluster)
    number_branches = num_branches(skeleton)
    skel, comp, comp_masks = split_skeleton_components(binary_cluster)
    all_branches = {}
    #  graph -> root -> branch tree -> attributes -> collect
    for mask in comp_masks:
        # m has the SAME shape as the original (no cropping), so coords are global
        G, coords = skeleton_to_csgraph(mask)
        coords = reshape_coords(coords)

        if len(coords) <= 5:
            continue

        root_graph = find_root_for_graph(G, coords)

        root_branch = create_branch_tree(G,coords, root_graph)

        if large_file:
            DFS_branch_tree(root_branch,remove_if_small_branch_extra)
        else:
            DFS_branch_tree(root_branch,remove_if_small_branch)

        DFS_branch_tree(root_branch, compute_brach_atributes)

        branches = designate_branches_map_to_coord(root_branch)
        
        all_branches.update(branches)     
    branches_map = assign_binary_to_branches(binary_cluster, all_branches)
    
    for br, pixels in branches_map.items():
        if br.length is None:
            br.compute_length()
        br.compute_thickness(pixels)
    
    full_branches_indice = [branches_map[b] for b in branches_map]

    # collect per-branch values (keep DFS/insertion order of branches_map)
    branches = list(branches_map.keys())
    # after you build `branches = list(branches_map.keys())`
    rows = []
    for b in branches:
        jy, jx = (b.all_coords[0] if getattr(b, "all_coords", None) else (np.nan, np.nan))
        if b.length < 5:
            continue
        # child endpoints around the junction (may be None)
        start_start_vec_child = getattr(b, "start_start_vec", None)   # (y,x)
        end_start_vec_child   = getattr(b, "end_start_vec",   None)   # (y,x)

        # parent endpoints around the junction – initialize to None first!
        parent_vector_around_trunck = None
        parent_vector_around_trunck   = getattr(b, "parent_vector_around_trunck",   None)

        rows.append(dict(
            branch_id=len(rows)+1,
            coords_0=(jy, jx),
            lengths=getattr(b, "length", np.nan),
            angle_from_parent_deg=getattr(b, "angle_from_parent", np.nan),
            thickness_px=getattr(b, "thickness", np.nan),
            num_branches=number_branches,

            # store endpoints as tuples or None
            start_start_vec_child=start_start_vec_child,
            end_start_vec_child=end_start_vec_child,
            parent_vector_around_trunck=parent_vector_around_trunck,
        ))

    df = pd.DataFrame(rows)
    if save:
        imsave(str(output_path), (skeleton.astype(np.uint8) * 255))
        csv_path = output_path.with_name(output_path.stem + "_branches_info.csv")
        df.to_csv(csv_path, index=False)
    if show:
        show_branches(binary_cluster, full_branches_indice)
    else:
        # Create a unique overlay file for each cluster
        overlay_path = output_path.with_name(output_path.stem + "_overlay.png")
        show_branches(binary_cluster, full_branches_indice, output_path=overlay_path, show=False)
    return df
    
def num_distinct(coords):
    # coords are (y, x); convert to hashable tuples and count uniques
    return len({(int(y), int(x)) for (y, x) in coords})

def find_root_of_skeleton(skeleton):
    skeleton_coords = np.where(skeleton > 0)
    if len(skeleton_coords[0]) == 0:
        return None
    max_y = np.max(skeleton_coords[0])
    bottom_pixels = np.where(skeleton_coords[0] == max_y)[0]
    x = skeleton_coords[1][bottom_pixels[len(bottom_pixels) // 2]]
    return (max_y, x)

def coord_of(i,coordinates): 
    y, x = tuple(coordinates[i]); 
    return y, x



# -------------------------
# Visualization helpers
# -------------------------
def show_branches(binary, branches, output_path=None, show=True):
    """
    Draw branches in distinct RGB colors over the binary image.
    Saves as a colored PNG if output_path is provided.
    """
    h, w = binary.shape
    # Start with grayscale base (3 channels)
    rgb_image = np.dstack([binary.astype(float)] * 3)

    np.random.seed(42)
    for branch in branches:
        color = np.random.rand(3)  # random RGB tuple
        for (y, x) in branch:
            if 0 <= y < h and 0 <= x < w:
                rgb_image[y, x, :] = color  # assign color directly (in RGB)

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_image)
        plt.title(f"Branches Visualization ({len(branches)} branches)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if output_path is not None:
        imsave(str(output_path), (rgb_image * 255).astype(np.uint8))  # true-color PNG

    return rgb_image

import re
import numpy as np
import matplotlib.pyplot as plt

def _parse_xy(v):
    """Accept (y,x) tuples/lists or '(y,x)' strings; return (y,x) or None."""
    if v is None:
        return None
    if isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2:
        y, x = float(v[0]), float(v[1])
        if np.isnan(y) or np.isnan(x):
            return None
        return (y, x)
    if isinstance(v, str):
        m = re.match(r"\s*\(?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)?\s*$", v)
        if m:
            y, x = float(m.group(1)), float(m.group(2))
            if np.isnan(y) or np.isnan(x):
                return None
            return (y, x)
    return None


def draw_angle_rays(image, df, ray_len_px=8, show_labels=True, out_path=None, dpi=200):
    """
    Draws:
      - Child rays (cyan) using start_start_vec_child → end_start_vec_child
      - Parent rays (yellow) using parent_vector_around_trunck + child start point as anchor
    """

    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    ax = plt.gca()

    def _plot_seg(p0, p1, color):
        (y0, x0), (y1, x1) = p0, p1
        ax.plot([x0, x1], [y0, y1], '-', lw=2, color=color)

    n_child, n_parent = 0, 0

    for i, r in df.iterrows():
        # -------------------- CHILD --------------------
        c0 = _parse_xy(r.get("start_start_vec_child"))
        c1 = _parse_xy(r.get("end_start_vec_child"))
        if c0 is not None and c1 is not None:
            _plot_seg(c0, c1, "cyan")
            n_child += 1

        # -------------------- PARENT --------------------
        parent_vec = _parse_xy(r.get("parent_vector_around_trunck"))
        if parent_vec is not None:
            a = c0 or c1
            if a is not None:
                dy, dx = parent_vec
                norm = np.hypot(dy, dx)
                if norm > 0:
                    dy *= ray_len_px / norm
                    dx *= ray_len_px / norm
                p1 = (a[0] + dy, a[1] + dx)
                _plot_seg(a, p1, "yellow")
                n_parent += 1

        # -------------------- LABELS --------------------
        if show_labels and c0 is not None:
            ax.text(c0[1], c0[0], str(i), fontsize=6, color="white",
                    ha="center", va="center")

    ax.set_axis_off()
    plt.tight_layout()

    print(f"[draw_angle_rays] drew {n_child} child rays and {n_parent} parent rays.")
    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()


def show_root(binary_cluster, coords, root_idx):
    """
    Quick plot: shows binary cluster and marks the root.
    """
    plt.imshow(binary_cluster, cmap='gray')
    y, x = coords[root_idx]  # coords are (row, col)
    plt.scatter(x, y, c='yellow', s=80, edgecolors='black')
    plt.title("Root position")
    plt.axis('off')
    plt.show()