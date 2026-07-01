import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.ndimage import convolve
from skimage.io import imsave
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.sparse import lil_matrix
from skimage.morphology import opening, closing, disk
from branch_tree import Branch
from typing import Tuple, List, Dict, Optional


# -------------------------
# Low-level helpers
# -------------------------
def get_nbrs(i, graph):
    #neighbors from CSR
    start, end = graph.indptr[i], graph.indptr[i + 1]
    return graph.indices[start:end]

def reshape_coords(coords):
    if isinstance(coords, tuple) and len(coords) == 2:
        coords = np.column_stack(coords).astype(int)  # shape (N, 2)
    return coords

def split_skeleton_components(binary_cluster, min_pixels=5):
    # binary_smooth = opening(binary_cluster, disk(4))
    # binary_smooth = closing(binary_smooth, disk(4))
    skel = skeletonize(binary_cluster.astype(bool))
    comp = label(skel, connectivity=2)          # 8-connected in 2D
    n = comp.max()
    masks = [(comp == i) for i in range(1, n+1)]
    return skel, comp, masks

def endpoints_from_skeleton(skeleton: np.ndarray):
    skel = skeleton.astype(bool)
    # 3x3 sum including the center pixel 
    K = np.ones((3, 3), dtype=np.uint8)
    sum3x3 = convolve(skel.astype(np.uint8), K, mode="constant", cval=0)
    # center==1 and exactly one 8-neighbor==1  -> sum3x3 == 2
    endpoints_mask = skel & (sum3x3 == 2)
    return endpoints_mask

def num_branches(skeleton):
    endpoints_mask = endpoints_from_skeleton(skeleton)
    return np.sum(endpoints_mask)

def coordinates_from_index(coordinates, index):
    y, x = tuple(coordinates[index])
    return y,x

def find_coord_index(coordinates,y_endpoint,x_endpoint):
    arr = np.asarray(coordinates, dtype=object)  # tolerate weird inputs
    for j in range(len(coordinates)):
        cy, cx = coordinates_from_index(coordinates,j)
        if cy == y_endpoint and cx == x_endpoint:
            return j
    return None

def find_root_for_graph(G, coords_rc):
    deg = np.diff(G.indptr)              # degree per node
    junctions = np.where(deg == 1)[0]
    if junctions.size:
        # bottommost junction (max row y)
        rid = junctions[np.argmax(coords_rc[junctions, 0])]
    else:
        # bottommost pixel
        rid = int(np.argmax(coords_rc[:, 0]))
    return tuple(coords_rc[rid])         # (y, x)



# -------------------------
# Spanning tree + branch extraction
# -------------------------

def find_spanning_tree(graph, root):
    """
    graph: CSR (preferably undirected). root: node index.
    Returns a CSR tree with exactly N-1 edges in the connected component.
    """
    G = graph.tocsr(copy=True)
    # ensure undirected (safe even if already is)
    if (G != G.T).nnz != 0:
        G = G.maximum(G.T)

    n = G.shape[0]
    tree = lil_matrix(G.shape, dtype=G.dtype)

    visited = set([root])
    stack = [root]

    while stack:
        u = stack.pop()
        for v in get_nbrs(u, G): 
            if v not in visited:
                visited.add(v)
                tree[u, v] = 1
                tree[v, u] = 1
                stack.append(v)

    return tree.tocsr()


def calc_subtree_lens(tree, root):
    """
    Iterative longest-subtree-lengths on a tree.
    Returns: dict {node: longest_length_to_any_descendant_leaf}
    """
    T = tree.tocsr()
    n = T.shape[0]

    # 1) Build parent/children with an iterative DFS
    parent = [-1] * n
    visited = set([root])
    stack = [root]
    children = [[] for _ in range(n)]

    while stack:
        u = stack.pop()
        for v in get_nbrs(u, T):
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            children[u].append(v)
            stack.append(v)

    # 2) Post-order over the explicit children lists (iterative)
    postorder = []
    st = [(root, 0)]  # (node, next child index)
    while st:
        u, idx = st.pop()
        if idx < len(children[u]):
            st.append((u, idx + 1))
            st.append((children[u][idx], 0))
        else:
            postorder.append(u)

    # 3) DP from leaves upward
    length = [0] * n
    for u in postorder:
        best = 0
        for v in children[u]:
            best = max(best, 1 + length[v])
        length[u] = best

    return {u: length[u] for u in visited}


def _unit_dir(a_rc, b_rc):
    vy = float(b_rc[0] - a_rc[0])
    vx = float(b_rc[1] - a_rc[1])
    n = (vy*vy + vx*vx) ** 0.5
    return (vy / n, vx / n) if n > 0 else None

def create_branches(graph, coordinates, root_index, subtree_lengths):
    visited = set()
    root_branch = Branch(None)
    stack = [(root_index, root_branch)]

    def coord_of(i):
        y, x = tuple(coordinates[i]); 
        return y, x

    while stack:
        i, branch = stack.pop()
        if i in visited:
            continue
        visited.add(i)

        branch.all_coords.append(coord_of(i))

        nbrs = [j for j in get_nbrs(i, graph) if j not in visited] 

        if len(nbrs) <= 1:
            for j in nbrs:
                stack.append((j, branch))
        else:
            keep_j = max(nbrs, key=lambda j: subtree_lengths.get(j, 0))
            # continue trunk on parent branch
            stack.append((keep_j, branch))
            # side branches become children
            for j in nbrs:
                if j == keep_j:
                    continue
                child = Branch(branch)
                # determine child parent vect by whats around trunck parent vec now
                parent_dir = None
                if len(branch.all_coords) >= 6:
                    p_prev = branch.all_coords[-6]        # five before
                    p_now  = branch.all_coords[-1]        # junction
                    parent_dir = _unit_dir(p_prev, p_now)
                elif len(branch.all_coords) >= 2:
                    p_prev = branch.all_coords[-2]        # one before
                    p_now  = branch.all_coords[-1]
                    parent_dir = _unit_dir(p_prev, p_now)
                else:
                    p_now = branch.all_coords[-1]         # just the junction available
                    parent_dir = _unit_dir(p_now, coord_of(keep_j))
                    
                child.parent_vector_around_trunck = parent_dir
                child.all_coords.append(coord_of(i))
                branch.add_child(child)
                stack.append((j, child))

    return root_branch

def create_branch_tree(graph,coordinates, root):
    # find root index (expects y,x)
    root_index = find_coord_index(coordinates, root[0], root[1])
    if root_index is None:
        # fallback: choose the closest node to the desired (y,x)
        arr = np.asarray(coordinates, dtype=int)
        dy = arr[:,0] - int(root[0]); dx = arr[:,1] - int(root[1])
        root_index = int(np.argmin(dy*dy + dx*dx))

    G_tree = find_spanning_tree(graph, root_index)
    lengths = calc_subtree_lens(G_tree, root_index)
    return create_branches(G_tree, coordinates, root_index, lengths)

# -------------------------
# Tree utilities and feature computation
# -------------------------

def DFS_branch_tree(root_branch,func):
    visited = set()
    def DFS_branch_tree_recursive(branch):
        func(branch)
        visited.add(branch)
        for child in branch.children:
            if child not in visited:
                DFS_branch_tree_recursive(child)

    DFS_branch_tree_recursive(root_branch)

def compute_brach_atributes(branch):
    branch.compute_start_vec()
    branch.compute_angle_with_parent()
    branch.compute_length()

def designate_branches_map_to_coord(root_branch):
    branches = {}
    def get_coords(branch):
        branches[branch] = list(branch.all_coords)
    DFS_branch_tree(root_branch, get_coords)
    return branches



def designate_pixel_to_branches(y,x, branches_map):
    best_branch = None
    best_d2 = float("inf")
    yi, xi = int(y), int(x)
    for br, coords in branches_map.items():
        if not coords:
            continue
        arr = np.asarray(coords, dtype=int)
        # ensure shape (N,2)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        elif arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        if arr.shape[1] > 2:
            arr = arr[:, :2]
        dy = arr[:, 0] - yi
        dx = arr[:, 1] - xi
        d2 = dy*dy + dx*dx
        m2 = float(d2.min())
        if m2 < best_d2:
            best_d2 = m2
            best_branch = br
    return best_branch


def assign_binary_to_branches(binary, branches_skel_map):
    branch_all_white_pixels_map = {}
    ys, xs = np.where(binary > 0)   # arrays of (y, x)
    for y, x in zip(ys, xs):
        br = designate_pixel_to_branches(y,x, branches_skel_map)
        if br is None:
            continue
        if br not in branch_all_white_pixels_map.keys():
            branch_all_white_pixels_map[br] = []
        branch_all_white_pixels_map[br].append((y,x))
    return branch_all_white_pixels_map

def unite_coords(parent, child):
    if parent and child:
        return parent + child[1:]



def remove_branch_from_parent(branch):
    branch_parent = branch.parent
    # re-parent grandchildren
    for branch_child in branch.children:
        branch_child.parent = branch_parent
    # rebuild parent's children without `branch`, keeping order
    new_children_of_parent = []
    for child in branch_parent.children:
        if child is branch:
            new_children_of_parent.extend(child.children) 
        else:
            new_children_of_parent.append(child)
    branch_parent.children = new_children_of_parent

def remove_if_small_branch_extra(branch):
    if branch.parent is None:
        return
    if len(branch.all_coords) > 15:
        return
    num_children = len(branch.children)
    if num_children > 1:
        return
    elif num_children == 1:
        child = branch.children[0]
        remove_branch_from_parent(child)
        branch.all_coords = unite_coords(branch.all_coords, child.all_coords)
    else:
        siblings = [sibling for sibling in (branch.parent).children if sibling != branch]
        remove_branch_from_parent(branch)
        if len(siblings) == 1:
            sibling = siblings[0]
            remove_branch_from_parent(sibling)
            (branch.parent).all_coords = unite_coords((branch.parent).all_coords, sibling.all_coords)

def remove_if_small_branch(branch):
    if branch.parent is None:
        return
    if len(branch.all_coords) > 8:
        return
    num_children = len(branch.children)
    if num_children > 1:
        return
    elif num_children == 1:
        child = branch.children[0]
        remove_branch_from_parent(child)
        branch.all_coords = unite_coords(branch.all_coords, child.all_coords)
    else:
        siblings = [sibling for sibling in (branch.parent).children if sibling != branch]
        remove_branch_from_parent(branch)
        if len(siblings) == 1:
            sibling = siblings[0]
            remove_branch_from_parent(sibling)
            (branch.parent).all_coords = unite_coords((branch.parent).all_coords, sibling.all_coords)

# -------------------------
# Simple stats
# -------------------------
def angle_stats(df, col="angle_from_parent_deg", zero_tol=0.0):
    """
    Return mean/std/count of angles (deg), ignoring NaN and values <= zero_tol.
    zero_tol lets you drop tiny numerics that are effectively 0.
    """
    a = pd.to_numeric(df.get(col), errors="coerce")
    a = a[(~a.isna()) & (a > zero_tol)]
    return (
        float(a.mean()) if len(a) else np.nan,
        float(a.std(ddof=1)) if len(a) > 1 else np.nan,
        int(len(a)),
        a,     # cleaned Series (for saving all angles if you want)
    )

