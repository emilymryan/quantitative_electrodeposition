# Standard library
import os
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from skimage import io, morphology, measure
from skimage.util import img_as_ubyte
from skimage.color import label2rgb, gray2rgb
from skimage.io import imsave
from scipy.ndimage import distance_transform_edt
from skimage import morphology

def label_clusters(binary, simulation):
    if simulation:
        labeled = measure.label(binary, connectivity=1)
        clean_labels = labeled * binary
        return clean_labels
    # Step 1: Remove small objects (area < 10)
    cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=10)
    imsave('cleaned.tif', (cleaned * 65535).astype(np.uint16))
    # Step 2: Dilate to merge components within 3 pixels distance
    dilated = morphology.binary_dilation(cleaned, morphology.disk(2))
    imsave('dilated.tif', (dilated * 65535).astype(np.uint16))
    # Step 3: Label connected regions
    labeled = measure.label(dilated, connectivity=1)
    clean_labels = labeled * binary
    return clean_labels

def save_clusters_to_images(clusters, output_dir="clusters"):
    os.makedirs(output_dir, exist_ok=True)
    for i, cluster in enumerate(clusters):
        filename = os.path.join(output_dir, f"cluster_{i}.png")
        cluster_img = cluster.astype(np.uint8) * 255 
        io.imsave(filename, img_as_ubyte(cluster_img))

    print(f"Saved {len(clusters)} clusters to '{output_dir}/'")

def get_clusters_from_labeled_img(labeled):
    clusters = []
    for region_label in np.unique(labeled):
        if region_label == 0:
            continue
        mask = (labeled == region_label)
        clusters.append(mask)
    return clusters
        

def get_cluster_height(cluster):
    white_pixels = np.argwhere(cluster == True)
    if white_pixels.size == 0:
        return 0  # No white pixels → height is 0
    # Extract y-coordinates (rows)
    y_coords = white_pixels[:, 0]
    height = y_coords.max() - y_coords.min()
    return height

def is_horizontal_bbox_mask(mask: np.ndarray, ratio_thresh: float = 0.20) -> bool:
    """Return True if a single 2-D mask looks like a horizontal bar by bbox aspect."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return False
    ys, xs = np.nonzero(m)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return (h / max(w, 1)) < ratio_thresh  # avoid div-by-zero

def get_cluster_height(mask: np.ndarray) -> int:
    """Height in pixels of the mask’s bbox."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return 0
    ys, _ = np.nonzero(m)
    return int(ys.max() - ys.min() + 1)

def separate_real_dendrites(clusters, threshold=60, simulation=False):
    real_dendrites, fake_dendrites = [], []

    for cluster in clusters:
        # validate
        if cluster is None or np.ndim(cluster) != 2:
            fake_dendrites.append(cluster)
            continue

        m = (cluster != 0)

        # filters for real images (not simulations)
        if not simulation:
            # “annotation area” heuristic: any ink in the last 6 rows
            is_annotation_area = np.any(m[-6:, :])  # note the ':'
            if is_annotation_area or is_horizontal_bbox_mask(m):
                fake_dendrites.append(cluster)
                continue

        # size / height filter
        cluster_height = get_cluster_height(m)
        if cluster_height < threshold:
            fake_dendrites.append(cluster)
        else:
            real_dendrites.append(cluster)

    return real_dendrites, fake_dendrites


def clusters_to_labeled_img(clusters):
    # Assume all cluster masks are the same shape
    shape = clusters[0].shape
    labeled = np.zeros(shape, dtype=np.int32)

    for idx, cluster in enumerate(clusters, start=1):
        labeled[cluster > 0] = idx

    return labeled


def save_labeled_img_hsv(labeled, original, path="hsv_labels.png"):
    max_label = labeled.max()
    h = np.linspace(0, 1, max_label + 1)
    colormap = hsv_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], axis=1))

    rgb_img = colormap[labeled]
    rgb_img[labeled == 0] = [0, 0, 0]

    mask = ~(np.all(rgb_img == [0, 0, 0], axis=-1))

    # Convert grayscale to RGB if needed
    if original.ndim == 2:
        on_original = gray2rgb(original)
    else:
        on_original = original.copy()

    # Normalize based on dynamic range
    max_val = on_original.max()
    if max_val > 255:
        on_original = on_original.astype(np.float32) / 65535.0
    elif max_val > 1.5:
        on_original = on_original.astype(np.float32) / 255.0
    else:
        on_original = on_original.astype(np.float32)

    # Overlay labels
    composite = on_original.copy()
    composite[mask] = rgb_img[mask]

    imsave(path, (composite * 255).astype(np.uint8))