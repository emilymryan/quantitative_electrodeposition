from preprocess import preprocess_image,preprocess_simulation_img_scale,is_large_file,dendrites_binary_gentle,sharpen_for_ocr
from clustering import *
import numpy as np
from skimage.io import imsave
from branches import branches_info,draw_angle_rays,show_branches
from branches_core import angle_stats
from scale import get_scale_data
from pathlib import Path
from info import row_white_ratio, frequency_of_big_dendrites,dendrite_height_width_ratios
from skimage import io  
import pandas as pd
from skimage.morphology import skeletonize

"""
for all the image we would like to output csv files with no meaning to each image whereabouts
1. over_all_aspect_ratio_std - compute all aspect ratios of all dendrites and then std
2. over_all_white_ratio_std - compute all row ratios of all dendrites and then std, by bins(any bin not having will be zero?)
3. over_all_thickness_std - compute all branches thicknesses of all dendrites and then std
4. over_all_angles_std - compute all branches angles of all dendrites and then std (removing no parent of course or 0)
5. over_all_dendrite_freq_std- each image to um freq and std
"""
from pathlib import Path
from skimage import util, color, exposure, filters

def image_info(img_path, simulation):
    img = io.imread(str(img_path))
    scale_img = img

    if simulation:
        img = preprocess_simulation_img_scale(img)
        scale_img = img

    # --- choose pipeline by file size ---
    large_file = is_large_file(img_path)

    if large_file:
        # normalize + CLAHE for robust dendrite binarization
        img_f = img.astype(np.float64, copy=False)
        s = np.percentile(img_f, 96) + 1e-8
        img_f = np.clip(img_f / s, 0, 1)
        img_clahe = exposure.equalize_adapthist(img_f, clip_limit=0.05)

        # gentle pipeline must return (binary, den)
        binary, den = dendrites_binary_gentle(img_clahe)

        # prepare grayscale uint8 for OCR/patterns
        ocr_u8, ocr_bin = sharpen_for_ocr(den)
        den = ocr_u8
        if den.ndim == 3:
            scale_img = util.img_as_ubyte(den)
        else:
            scale_img = util.img_as_ubyte(util.img_as_float(den))

        # binary for scale detection (Otsu)
        th = filters.threshold_otsu(scale_img)
        binary_scale = scale_img > th

        thr_real = 100  # stricter for large
    else:
        # small/clean path
        binary = preprocess_image(img, simulation)
        binary_scale = binary

        # make sure scale_img is cv2/uint8 friendly
        scale_img = img
        if scale_img.dtype != np.uint8:
            scale_img = util.img_as_ubyte(scale_img)
        if scale_img.ndim == 3 and scale_img.shape[2] == 4:  # RGBA→RGB
            scale_img = util.img_as_ubyte(color.rgba2rgb(scale_img))

        thr_real = 60

    # --- scale info using scale_img + binary_scale ---
    scale_info = get_scale_data(scale_img, binary_scale, simulation)
    scale_um_per_px = float(scale_info["scale"]) if scale_info else np.nan

    # defaults so we always return safely
    freq_big_dendrites = np.nan
    dendrites_height_to_width_ratio = pd.Series(dtype=float)
    row_whites = None
    all_branches_thickness = []
    all_branches_angles    = []
    all_branch_lengths     = []
    all_num_branches       = []


    if scale_info is not None:
        # cluster & filter "real" dendrites
        labeled  = label_clusters(binary, simulation)
        clusters = get_clusters_from_labeled_img(labeled)
        real, _  = separate_real_dendrites(clusters, threshold=thr_real, simulation=simulation)
        num_dends = len(real)

        # per-image frequency
        freq_big_dendrites = frequency_of_big_dendrites(scale_info, num_dends)

        # aspect ratios
        base = Path(img_path).stem if hasattr(img_path, "stem") else Path(str(img_path)).stem
        df_hw = dendrite_height_width_ratios(real, um_per_px=scale_um_per_px, image_name=base)
        if isinstance(df_hw, pd.DataFrame) and "height_to_width_ratio" in df_hw.columns:
            dendrites_height_to_width_ratio = pd.to_numeric(
                df_hw["height_to_width_ratio"], errors="coerce"
            ).dropna()

        # white ratio profile
        positions_um, ratios_10nm, *_ = row_white_ratio(simulation, binary, scale_um_per_px)
        row_whites = pd.DataFrame({"position_um": positions_um, "ratio": ratios_10nm})
        # per-branch collections (convert px→µm where relevant)
        for i, cluster in enumerate(real):
            df_branch = branches_info(cluster, Path("info"),large_file, False, False)
            if isinstance(df_branch, pd.DataFrame) and not df_branch.empty:
                if "thickness_px" in df_branch.columns:
                    vals = pd.to_numeric(df_branch["thickness_px"], errors="coerce").dropna()
                    all_branches_thickness.extend((vals * scale_um_per_px).tolist())
                if "angle_from_parent_deg" in df_branch.columns:
                    vals = pd.to_numeric(df_branch["angle_from_parent_deg"], errors="coerce").dropna()
                    all_branches_angles.extend(vals.tolist())
                if "lengths" in df_branch.columns:
                    vals = pd.to_numeric(df_branch["lengths"], errors="coerce").dropna()
                    all_branch_lengths.extend((vals * scale_um_per_px).tolist())
                if "num_branches" in df_branch.columns:
                    vals = pd.to_numeric(df_branch["num_branches"], errors="coerce").dropna()
                    all_num_branches.extend(vals.tolist())

    return (
        freq_big_dendrites,
        dendrites_height_to_width_ratio,
        row_whites,
        all_branches_thickness,
        all_branches_angles,
        all_branch_lengths,
        all_num_branches,
    )

def procces_images_to_csv(image_paths,simulation):
    all_freq_dendrites = []
    all_height_width = []
    all_row_whites = []
    all_thickness = []
    all_angles = []
    all_lengths = []
    all_num_branches = []
    for img_path in image_paths:
        (freq_dend,
         height_width,
         row_whites,
         b_thickness,
         b_angles,
         b_lengths,
         num_branches) = image_info(img_path, simulation)

        all_row_whites.append(row_whites)

        if isinstance(freq_dend, (list, np.ndarray, pd.Series)):
            all_freq_dendrites.extend(freq_dend)
        else:
            all_freq_dendrites.append(freq_dend)

        if isinstance(height_width, (list, np.ndarray, pd.Series)):
            all_height_width.extend(height_width)
        else:
            all_height_width.append(height_width)

        if isinstance(b_thickness, (list, np.ndarray, pd.Series)):
            all_thickness.extend(b_thickness)
        else:
            all_thickness.append(b_thickness)

        if isinstance(b_angles, (list, np.ndarray, pd.Series)):
            all_angles.extend(b_angles)
        else:
            all_angles.append(b_angles)

        if isinstance(b_lengths, (list, np.ndarray, pd.Series)):
            all_lengths.extend(b_lengths)
        else:
            all_lengths.append(b_lengths)

        if isinstance(num_branches, (list, np.ndarray, pd.Series)):
            all_num_branches.extend(num_branches)
        else:
            all_num_branches.append(num_branches)

    pooled_row_whites = (
        pd.concat(
            [df for df in all_row_whites if isinstance(df, pd.DataFrame) and not df.empty],
            ignore_index=True
        )
        if all_row_whites else pd.DataFrame(columns=["position_um","ratio"])
    )

    return {
        "freq_dendrites": all_freq_dendrites,
        "row_whites":     pooled_row_whites,   # <- return a DataFrame, not a list
        "aspect_ratios":  all_height_width,
        "thickness":      all_thickness,
        "angles":         all_angles,
        "lengths":        all_lengths,
        "num_branches":   all_num_branches
    }

def row_whites_info(df_row_whites):
    """
    Groups the white ratio data by position_um and computes
    mean, std, and count for each bin, preserving position_um.
    """
    if df_row_whites is None or df_row_whites.empty:
        return pd.DataFrame(columns=["position_um", "ratio_mean", "ratio_std", "n_points"])

    # Make sure numeric
    df_row_whites["position_um"] = pd.to_numeric(df_row_whites["position_um"], errors="coerce")
    df_row_whites["ratio"] = pd.to_numeric(df_row_whites["ratio"], errors="coerce")

    # Group and compute stats
    grouped = (
        df_row_whites.groupby("position_um", as_index=False)
        .agg(ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            n_points=("ratio", "count"))
        .sort_values("position_um")
        .reset_index(drop=True)
    )

    # Ensure explicit column for position_um
    grouped["position_um"] = grouped["position_um"]

    return grouped

    

def write_overall_csvs(image_paths, simulation, output_dir="info",fill_empty_bins_with_zero=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled = procces_images_to_csv(image_paths, simulation)

    # 1) Aspect ratios (per-dendrite)
    aspect = pd.to_numeric(pd.Series(pooled["aspect_ratios"]), errors="coerce").dropna()
    aspect_std = float(aspect.std(ddof=1)) if len(aspect) > 1 else np.nan
    pd.DataFrame({
        "aspect_ratio": aspect,
        "overall_std":  np.full(len(aspect), aspect_std)
    }).to_csv(output_dir / "all_aspect_ratios_with_std.csv", index=False)

    # 2)row whites save to csv
    df_row_whites = pooled["row_whites"]
    df_white_stats = row_whites_info(df_row_whites)
    df_white_stats.to_csv(output_dir / "row_white_ratios_grouped_with_std.csv", index=False)

    # 3) Thickness (per-branch)
    thick = pd.to_numeric(pd.Series(pooled["thickness"]), errors="coerce").dropna()
    thick_std = float(thick.std(ddof=1)) if len(thick) > 1 else np.nan
    pd.DataFrame({
        "thickness":   thick,
        "overall_std": np.full(len(thick), thick_std)
    }).to_csv(output_dir / "all_branch_thickness_with_std.csv", index=False)

    # 4) Angles (per-branch; assume zeros already filtered upstream; if not, drop them here)
    ang = pd.to_numeric(pd.Series(pooled["angles"]), errors="coerce").dropna()
    ang = ang[ang != 0]
    ang_std = float(ang.std(ddof=1)) if len(ang) > 1 else np.nan
    pd.DataFrame({
        "angle_deg":   ang,
        "overall_std": np.full(len(ang), ang_std)
    }).to_csv(output_dir / "all_branch_angles_with_std.csv", index=False)

    # 5) Dendrite frequency per image
    freq = pd.to_numeric(pd.Series(pooled["freq_dendrites"]), errors="coerce").dropna()
    freq_std = float(freq.std(ddof=1)) if len(freq) > 1 else np.nan
    pd.DataFrame({
        "freq":        freq,
        "overall_std": np.full(len(freq), freq_std)
    }).to_csv(output_dir / "dendrite_freq_per_image_with_std.csv", index=False)

    # (Optional) return the stds if you still want them programmatically
    return {
        "aspect_ratio_std": aspect_std,
        "thickness_std":    thick_std,
        "angles_std":       ang_std,
        "freq_std":         freq_std
    }
