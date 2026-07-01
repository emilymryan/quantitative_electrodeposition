from preprocess import preprocess_image,preprocess_simulation_img_scale,dendrites_binary_gentle,is_large_file,sharpen_for_ocr
from clustering import *
#import numpy as np
from skimage.io import imsave
from branches import branches_info,draw_angle_rays
from branches_core import angle_stats
from scale import get_scale_data
from pathlib import Path
from info import row_white_ratio, frequency_of_big_dendrites,dendrite_height_width_ratios
from skimage import io  
import pandas as pd
from skimage.morphology import skeletonize
from skimage.morphology import opening, closing
from csv_info import write_overall_csvs
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk
from skimage.filters import median
# Third-party libraries
#import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, morphology, measure, util
from skimage.color import rgb2gray,rgba2rgb

import numpy as np
from skimage import io, color, filters, morphology, restoration, exposure, util
from skimage.morphology import disk
import numpy as np
from skimage import io, color, filters, morphology, restoration, util, exposure
from skimage.morphology import disk
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from skimage import exposure, util



def process_img(img_path, simulation):

    img = io.imread(img_path)
    print(f"Loaded image from: {img_path}")
    scale_img = img
    if simulation:
        img = preprocess_simulation_img_scale(img)
        scale_img = img
    large_file = is_large_file(img_path)
    if large_file:
        img = img.astype(np.float64)
        # Normalize using percentile to avoid single-pixel overdominance
        scale = np.percentile(img, 96) + 1e-8
        img /= scale
        img = np.clip(img, 0, 1)
        img_clahe = exposure.equalize_adapthist(img, clip_limit=0.05)
        binary, den = dendrites_binary_gentle(img_clahe)
        ocr_u8, ocr_bin = sharpen_for_ocr(den)
        den = ocr_u8
        if den.ndim == 3:
            # if it’s already uint8, this is a no-op
            scale_img = util.img_as_ubyte(den)
        else:
            # grayscale float -> uint8
            scale_img = util.img_as_ubyte(util.img_as_float(den))
        from skimage.filters import threshold_otsu

        # Convert grayscale to binary
        th = threshold_otsu(scale_img)
        binary_scale = scale_img > th
    else:
        binary = preprocess_image(img,simulation)
        binary_scale = binary
    # turn image to binary
    # get scale info in format of microns,pixel length , scale, coords
    scale_info = get_scale_data(scale_img,binary_scale,simulation)
    # Label connected regions
        
    labeled = label_clusters(binary,simulation)
    clusters = get_clusters_from_labeled_img(labeled)
    # possibility to save more with threashold height and remove info bar
    threshold = 40 # changed from 60 to 40
    if large_file:
        threshold = 85
    real, _ = separate_real_dendrites(clusters, threshold=threshold, simulation=simulation)

    num_dendrites = len(real)
    #not nessecary
    save_clusters_to_images(real)
    labeled = clusters_to_labeled_img(real)
    # Make sure output folder exists
    output_dir = Path("info")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Loop through all clusters
    for i, cluster in enumerate(real):
        output_path = output_dir / f"info{i}.tif"
        df = branches_info(cluster, output_path,large_file,False,True)
        #draw_angle_rays(skel, df, out_path="angles_overlay.png")
    # colorful picture of clusters
    save_labeled_img_hsv(labeled,binary)
    if scale_info is None or "scale" not in scale_info or scale_info["scale"] is None:
        print("[process_img] No scale detected → skipping row_white_ratio + freq_big_dendrites.")
        positions_um = np.array([])
        ratios_10nm  = np.array([])
        n_rows       = np.array([])
        mean_ratio   = np.nan
        freq_big     = np.nan
    else:
        try:
            um_per_px = float(scale_info["scale"])
        except Exception:
            print(f"[process_img] Bad scale value in scale_info={scale_info} → skipping density.")
            um_per_px = None

        if um_per_px is None or not np.isfinite(um_per_px):
            print("[process_img] Invalid um_per_px → skipping row_white_ratio + freq_big_dendrites.")
            positions_um = np.array([])
            ratios_10nm  = np.array([])
            n_rows       = np.array([])
            mean_ratio   = np.nan
            freq_big     = np.nan
        else:
            # compute density safely
            positions_um, ratios_10nm, mean_ratio, n_rows = row_white_ratio(simulation,binary, um_per_px)
            print("white (row density ratios):", ratios_10nm)

            # frequency of big dendrites needs a valid scale_info too
            try:
                freq_big = frequency_of_big_dendrites(scale_info, num_dendrites)
            except Exception as e:
                print(f"[process_img] frequency_of_big_dendrites failed: {e}")
                freq_big = np.nan

    print("freq big dendrites:", freq_big)
    

def process_folder(file_path,simulation):
    # Build list of images
    file_path = Path(file_path)
    if file_path.is_dir():
        exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_paths = sorted(p for p in file_path.iterdir() if p.suffix.lower() in exts)
    else:
        image_paths = [file_path]

    output_dir = Path("info")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_overall_csvs(image_paths, simulation, output_dir,True)

#process_img("/Users/noajacobs/Documents/dendrite_analysis_project/Ag_2e-9_003.tif",False)
process_img("C:\\Users\\epens\\Ag Electrodeposition Paper\\Aug27_1e-4RR_0.23V_0.42s_bw.png",True)
#process_img('/Users/noajacobs/Documents/dendrite_analysis_project/simulation_file/0.22V_1E-4RR_0.98s_9-12.png',True)
#process_folder("/Users/noajacobs/Documents/dendrite_analysis_project/dificult_mages",False)

# output_dir = Path("/Users/noajacobs/Documents/dendrite_analysis_project")
output_dir = Path("C:\\Users\\epens\\Ag Electrodeposition Paper")
csv_path = output_dir / "info.csv"              # join dir + filename
#summary_df.to_csv(csv_path, index=False)
print(f"wrote {csv_path}")

process_folder("C:\\Users\\epens\\Ag Electrodeposition Paper",True)
#process_folder("/Users/noajacobs/Documents/dendrite_analysis_project/simulation_file",True)

# output_dir = Path("/Users/noajacobs/Documents/dendrite_analysis_project")
# csv_path = output_dir / "simulation_info.csv"              # join dir + filename
# summary_df.to_csv(csv_path, index=False)
# print(f"wrote {csv_path}")
# Build your image list with the same logic you used before