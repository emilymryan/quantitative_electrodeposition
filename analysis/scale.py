# # Third-party libraries
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import re
from skimage.util.shape import view_as_windows
SIZE_OF_IMAGE_MICRONS = 10
UNITS_FOR_MICRONS = ['μm','um','pm','nm','tm','ym','ηπι','µπι']
PATTERNS_FOR_SCALE_LEFT = [
        [
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ],
    [
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ],
    # L3: side thickness = 2 cols → 4 cols, bar = 3 rows
    [
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ],
        [
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ],
    [
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]
]
PATTERNS_FOR_SCALE_RIGHT = [
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0],
    ],
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
    ],
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
    ],
        [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
    ],
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
    ]
]
# ---- turn your python lists into numpy arrays (0/1, shape (N,10,10)) ----
def _to_np_patterns(lst_of_lst):
    pats = [np.array(p, dtype=np.uint8) for p in lst_of_lst]
    return np.stack(pats, axis=0)

def _match_one(img_bin, template):
    tpl = np.array(template, dtype=np.uint8)
    th, tw = tpl.shape
    # build windows sized like the template
    wins = view_as_windows(img_bin, (th, tw))   # -> (H-th+1, W-tw+1, th, tw)
    # compare and score
    score = (wins == tpl).mean(axis=(2, 3))     # fraction match
    return (score >= 0.97)

def _inner_left_edge_col(tpl):
    tpl = np.asarray(tpl, dtype=np.uint8)
    # pick the row(s) with the long horizontal bar (max row sum)
    row = np.argmax(tpl.sum(axis=1))
    # first 0→1 transition = inner edge
    r = tpl[row]
    for c in range(1, r.size):
        if r[c-1] == 0 and r[c] == 1:
            return c
    return 0  # fallback

def _inner_right_edge_col(tpl):
    tpl = np.asarray(tpl, dtype=np.uint8)
    row = np.argmax(tpl.sum(axis=1))
    r = tpl[row]
    # last 1→0 transition = inner edge
    for c in range(r.size-1, 0, -1):
        if r[c-1] == 1 and r[c] == 0:
            return c-1
    return r.size-1  # fallback

# ---- main finder ----
def find_scale_endcaps_and_distance(binary, PATTERNS_FOR_SCALE_LEFT, PATTERNS_FOR_SCALE_RIGHT, same_row_tol=0):
    """
    binary: 2D array of 0/1
    PATTERNS_FOR_SCALE_LEFT/RIGHT: lists/arrays of templates (each h x w) with 0/1
    Returns list of dicts with y, x_left, x_right, left_edge_x, right_edge_x, distance_pixels, idx_left, idx_right
    """
    from skimage import io, img_as_ubyte
    import numpy as np

    # Suppose your binary image is stored in variable `binary`
    # (values 0/1 or True/False)

    bin01 = (binary > 0).astype(np.uint8)
    
    # Precompute bar widths (if you have helpers, keep using them)
    def _left_bar_width(tpl):
        tpl = np.asarray(tpl, dtype=np.uint8)
        # width of the left vertical bar = count of ones in the first nonzero columns from the left
        colsum = tpl.sum(axis=0)
        nz = np.where(colsum > 0)[0]
        # contiguous run from the left edge
        if nz.size == 0: return 0
        first = nz[0]
        run = 1
        for c in range(first+1, tpl.shape[1]):
            if colsum[c] > 0: run += 1
            else: break
        return run

    def _right_bar_width(tpl):
        tpl = np.asarray(tpl, dtype=np.uint8)
        # width of right vertical bar = count of ones in the last nonzero columns from the right
        colsum = tpl.sum(axis=0)
        nz = np.where(colsum > 0)[0]
        if nz.size == 0: return 0
        last = nz[-1]
        run = 1
        for c in range(last-1, -1, -1):
            if colsum[c] > 0: run += 1
            else: break
        return run

    left_bar_w  = [ _left_bar_width(PATTERNS_FOR_SCALE_LEFT[i])   for i in range(len(PATTERNS_FOR_SCALE_LEFT)) ]
    right_bar_w = [ _right_bar_width(PATTERNS_FOR_SCALE_RIGHT[i]) for i in range(len(PATTERNS_FOR_SCALE_RIGHT)) ]

    # ---- find all LEFT hits
    left_hits = []  # (y, x, idxL, tpl_w)
    for idxL, tplL in enumerate(PATTERNS_FOR_SCALE_LEFT):
        mL = _match_one(bin01, tplL)   # (H-h+1, W-w+1) bool
        ys, xs = np.where(mL)
        if ys.size:
            tw = np.asarray(tplL).shape[1]
            left_hits.extend(zip(ys.tolist(), xs.tolist(), [idxL]*len(ys), [tw]*len(ys)))

    # ---- find all RIGHT hits
    right_hits_by_y = {}  # y -> list of (x, idxR, tpl_w)
    for idxR, tplR in enumerate(PATTERNS_FOR_SCALE_RIGHT):
        mR = _match_one(bin01, tplR)
        ys, xs = np.where(mR)
        if ys.size:
            tw = np.asarray(tplR).shape[1]
            for y, x in zip(ys.tolist(), xs.tolist()):
                right_hits_by_y.setdefault(y, []).append((x, idxR, tw))

    # ---- pair L/R on (approximately) the same row
    results = []
    for yL, xL, idxL, twL in left_hits:
        # allow small tolerance in y if needed
        candidate_rows = [yL]
        if same_row_tol > 0:
            candidate_rows = range(max(0, yL - same_row_tol), yL + same_row_tol + 1)

        # gather right candidates on same row(s) with xR > xL
        candidates = []
        for yy in candidate_rows:
            if yy in right_hits_by_y:
                for (xR, idxR, twR) in right_hits_by_y[yy]:
                    if xR > xL:
                        candidates.append((yy, xR, idxR, twR))

        if not candidates:
            continue

        # nearest right on x
        yR, xR, idxR, twR = min(candidates, key=lambda t: t[1])

        # Precompute these once for all templates
        left_inner_offsets  = [ _inner_left_edge_col(t)  for t in PATTERNS_FOR_SCALE_LEFT ]
        right_inner_offsets = [ _inner_right_edge_col(t) for t in PATTERNS_FOR_SCALE_RIGHT ]

        # When pairing left and right hits:
        left_edge_x  = xL + left_inner_offsets[idxL]
        right_edge_x = xR + right_inner_offsets[idxR]

        distance = right_edge_x - left_edge_x
        results.append({
            "y": int(yL),
            "x_left": int(xL),
            "x_right": int(xR),
            "left_edge_x": int(left_edge_x),
            "right_edge_x": int(right_edge_x),
            "distance_pixels": int(distance),
            "idx_left": int(idxL),
            "idx_right": int(idxR),
        })
    return results

def display_scale(img,left_edge_x,right_edge_x,mid_y):
    # ---- draw the bar on the original image and display/save ----
    H, W = img.shape[:2]
    # 1) make a safe RGB uint8 copy for drawing
    if img.ndim == 2:
        vis = img.copy()
        # normalize dtype
        if vis.dtype != np.uint8:
            if vis.max() <= 1.0:  # float in 0..1 or bool
                vis = (vis.astype(np.float32) * 255).clip(0,255).astype(np.uint8)
            else:
                vis = vis.astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)   # RGB
    else:
        vis = img.copy()
        # skimage gives RGB; OpenCV draws in BGR but we’ll *display* with plt (RGB), so keep RGB
        if vis.dtype != np.uint8:
            if vis.max() <= 1.0:
                vis = (vis.astype(np.float32) * 255).clip(0,255).astype(np.uint8)
            else:
                vis = vis.astype(np.uint8)

    # 2) clamp coordinates
    lx = max(0, min(W-1, int(left_edge_x)))
    rx = max(0, min(W-1, int(right_edge_x)))
    my = max(0, min(H-1, int(mid_y)))

    # 3) draw a red line (RGB)
    cv2.line(vis, (lx, my), (rx, my), (255, 0, 0), 2)

    # 4) show (matplotlib expects RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(vis)
    plt.axis('off')
    plt.show()


def extract_single_number(text):
    if not text:
        return None
    
    matches = re.findall(r'\d+', text)
    if len(matches) == 1:   # exactly one group of digits
        return int(matches[0])
    return None

def find_microns(data):
    nm_flag = False
    units = UNITS_FOR_MICRONS
    microns = None
    for i, word in enumerate(data['text']):
        w = (word or "").lower()
        if any(u in w for u in units):
            if 'nm' in w:
                nm_flag = True
            prev_text = data['text'][i-1] if i > 0 else "" 
            try:
                numbers_only = extract_single_number(prev_text)
                microns = float(numbers_only)
                if nm_flag or microns > 50:
                    microns = microns/1000
                    break
            except:
                continue
    return microns

def get_scale_data(img, binary, simulation):
    # Step 1: Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.astype(np.uint8)

    if simulation:
        H, W = img.shape[:2]
        microns = float(SIZE_OF_IMAGE_MICRONS)          # whole-image height in µm
        pixels  = int(H)                                 # whole-image height in px
        return {
            "microns": microns,
            "pixels":  pixels,
            "scale":   microns / float(pixels),          # µm per pixel
            "bar_coords": None,                          # no real bar; that’s fine
        }
    # Step 2: OCR to get positions
    data = pytesseract.image_to_data(
    gray,
    output_type=pytesseract.Output.DICT,
    lang="eng+ell")
    microns = find_microns(data)
    if microns is not None:
        L = _to_np_patterns(PATTERNS_FOR_SCALE_LEFT)
        R = _to_np_patterns(PATTERNS_FOR_SCALE_RIGHT) 
        # binary is your 0/1 mask (np.uint8 or bool)
        hits = find_scale_endcaps_and_distance(binary, L, R)
        if hits:
            h0 = hits[0]
            y  = h0["y"] + 5                  
            lx = h0["left_edge_x"]
            rx = h0["right_edge_x"]
            pixel_length = rx - lx               

            # visualize on the ORIGINAL image
            display_scale(binary, lx, rx, y)

            # return scale info
            return {
                "microns": microns,
                "pixels":  int(pixel_length),
                "scale":   float(microns) / float(pixel_length),  # µm per pixel
                "bar_coords": [[int(lx), int(y)], [int(rx), int(y)]],
            }
    return None
