# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, morphology, measure, util
from skimage.color import rgb2gray,rgba2rgb
from clustering import *
import numpy as np
from skimage.io import imsave
from scale import get_scale_data
from pathlib import Path
from info import row_white_ratio, frequency_of_big_dendrites,dendrite_height_width_ratios
from skimage import io  
import pandas as pd
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk
from skimage.filters import median
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, morphology, measure, util
from skimage.color import rgb2gray,rgba2rgb
from skimage import io, color, filters, morphology, restoration, exposure, util
from skimage.morphology import disk
import numpy as np
from skimage import io, color, filters, morphology, restoration, util, exposure
from skimage.morphology import disk
import matplotlib.pyplot as plt

import cv2
import numpy as np
from skimage import exposure, util

def sharpen_for_ocr(den):
    """Make den (float [0,1]) sharper and OCR-friendly."""
    g = util.img_as_ubyte(np.clip(den, 0, 1))                # → uint8
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = exposure.equalize_adapthist(g, clip_limit=0.015)      # stronger contrast
    g = util.img_as_ubyte(g)
    blur = cv2.GaussianBlur(g, (0, 0), 1)                    # unsharp mask
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    _, g_bin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return g, g_bin                                           # uint8 gray + binary
                                     # uint8 grayscale, uint8 binary
def is_large_file(path, size_mb_threshold=5):
    """Return True if file is larger than the threshold in megabytes."""
    path = Path(path)
    if not path.exists():
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    return size_mb > size_mb_threshold

def dendrites_binary_gentle(img,
                            crop_top=0.18,
                            crop_bottom=0.12,
                            sigma_color=0.02,  # gentler than 0.05
                            sauvola_w=41,
                            sauvola_k=0.10,
                            min_obj=120,
                            min_hole=100):
    # 1) grayscale to [0,1]
    gray = color.rgb2gray(img) if img.ndim == 3 else util.img_as_float(img)

    H, W = gray.shape
    roi = np.ones_like(gray, bool)
    if crop_top:
        roi[:int(H * crop_top)] = False  # remove “sky”
    if crop_bottom:
        roi[-int(H * crop_bottom):] = False  # remove info bar

    # 2) mild edge-preserving denoise
    den = restoration.denoise_bilateral(gray, sigma_color=sigma_color,
                                        sigma_spatial=2, channel_axis=None)

    # 3) light local contrast (helps thin branches pop)
    den = exposure.equalize_adapthist(den, clip_limit=0.005)  # very small
    # 4) adaptive threshold (Sauvola)
    th = filters.threshold_sauvola(den, window_size=sauvola_w, k=sauvola_k)
    
    b = (den > th) & roi
    # 5) tiny morphology to remove pepper/salt delicately
    b = morphology.binary_opening(b, footprint=disk(1))  # removes single-pixel whiskers
    b = morphology.remove_small_objects(b, min_size=min_obj)
    b = morphology.remove_small_holes(b, area_threshold=min_hole)

    return b, den

def preprocess_simulation_img_scale(img):
    if img.shape[-1] != 4:
        raise ValueError("Expected RGBA image")
    alpha = img[..., 3]
    # mask of non-transparent pixels
    mask = alpha > 0
    if not mask.any():
        raise ValueError("Image has no opaque content")
    # find bounding box of non-transparent region
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    # crop only to that region
    rgb_cropped = img[y0:y1, x0:x1, :3]
    rgb_cropped = rgb_cropped[:-8, :, :] # changed from 8 pixels to 10
    # convert to grayscale
    gray = rgb2gray(rgb_cropped)
    return gray

def preprocess_image(img,simulation):
    if img.ndim == 2 or simulation:  # already grayscale
            pass
    elif img.ndim == 3:
            img = rgb2gray(img)
    else:
        raise ValueError(f"Unsupported channel count: {img.shape[-1]}")
    # cast
    img = img.astype(np.float64)
    # Normalize using percentile to avoid single-pixel overdominance
    scale = np.percentile(img, 99) + 1e-8
    img /= scale
    img = np.clip(img, 0, 1)

    # Apply gentle CLAHE
    img_clahe = exposure.equalize_adapthist(img, clip_limit=0.005)

    # threshold
    from skimage.filters import threshold_local
    thresh = threshold_local(img_clahe, block_size=51, offset=-0.1)
    binary = img_clahe > thresh
    return binary

