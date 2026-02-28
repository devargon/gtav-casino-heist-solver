import os

import numpy as np
import cv2
from PIL import Image, ImageGrab, ImageDraw  # Windows/macOS; for Linux use mss

def tile_feat(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (1,1), 0)
    # normalize contrast so gray vs white differences matter less
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return g

def preprocess_game_fingerprint(bgr: np.ndarray) -> np.ndarray:
    # 1) Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2) Contrast normalization (fixed settings)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) Remove vertical dotted grid
    # Tall, skinny kernel tuned for ~500px-tall ROI
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
    vertical_texture = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

    degrid = cv2.subtract(gray, vertical_texture)

    # 4) Extract ridge structures by scale
    ridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(degrid, cv2.MORPH_OPEN, ridge_kernel)

    ridges = cv2.subtract(degrid, opened)

    # 5) Binary ridge mask
    _, binary = cv2.threshold(ridges, 30, 255, cv2.THRESH_BINARY)

    # 6) Minimal cleanup
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8), iterations=1)

    return binary

def grab_fingerprint_from_game(screenshot_file):
    # x,y,w,h in screen coordinates
    # img = ImageGrab.grab(bbox=(x, y, x+w, y+h))  # returns PIL Image (RGB)
    print("tst")
    img = Image.open(screenshot_file)
    img = img.resize((1920, 1080))

    # crop dimensions, top left: 979, 154, selection dimensions: 338, 507
    img = img.crop((979, 154, 979+338, 154+507))
    arr = np.array(img)                          # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    result = preprocess_game_fingerprint(bgr)
    return bgr

def grab_options_from_game(screenshot_file):
    img = Image.open(screenshot_file)
    img = img.resize((1920, 1080))

    data = []

    top_left = (472, 267)
    option_length = 125
    option_gap = 19

    x, y = top_left

    for i in range(8):
        cropped = img.crop((x, y, x + option_length, y + option_length)).resize((114, 114))
        arr = np.array(cropped)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        data.append({"bbox": (x, y, x + option_length, y + option_length), "np": bgr})
        if i % 2 == 0:
            x += option_length + option_gap
        else:
            x = top_left[0]
            y += option_length + option_gap
        cropped.close()
    img.close()

    return data

# used to detect the correct main fingerprint
def score_orb_homography(live, ref, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)

    k1, d1 = orb.detectAndCompute(live, None)
    k2, d2 = orb.detectAndCompute(ref, None)

    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        return 0

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if mask is None:
        return 0

    inliers = int(mask.ravel().sum())
    return inliers

# simpler and faster method to compare tiles
def match_fp_option_tile(live_bgr, ref_bgr):
    live = tile_feat(live_bgr)
    ref  = tile_feat(ref_bgr)

    # if sizes differ, resize reference to live (or vice versa)
    ref = cv2.resize(ref, (live.shape[1], live.shape[0]))

    # try both polarities
    live_inv = 255 - live

    s1 = cv2.matchTemplate(live,     ref, cv2.TM_CCOEFF_NORMED).max()
    s2 = cv2.matchTemplate(live_inv, ref, cv2.TM_CCOEFF_NORMED).max()

    return max(float(s1), float(s2))


if __name__ == "__main__":
    screenshot_filename = "sample_screenshot.jpg"
    print("Detecting fingerprint to solve in game.")
    live = grab_fingerprint_from_game(screenshot_filename)
    print("Detecting available fingerprint tile options in game.")
    options_data = grab_options_from_game(screenshot_filename)
    # [ { bbox: (x1,y1,x2,y2), np: np.array }, ... ]
    best_name = None
    best_score = -1
    best_projection = None

    for fn in os.listdir("reference_fp"):
        print(f"Comparing fingerprint against {fn}.")
        ref = cv2.imread(os.path.join("reference_fp", fn), cv2.IMREAD_COLOR)
        if ref is None:
            continue
        ref_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)
        ref_gray = ref_bgr

        # ref_gray = preprocess_game_fingerprint(ref)
        score = score_orb_homography(live, ref_gray)

        print(f"{fn} returned {score} inliers")

        if score > best_score:
            print(f"New best match: {fn}")
            best_score = score
            best_name = fn
            best_projection = ref_gray

    print(f"Best match: {best_name} with {best_score} inliers")
    answer_name =  best_name.split(".")[0]
    answer_directory = os.path.join("fp_answers", answer_name)
    correct_answer_arrays = []
    print(f"Loading correct answer options from {answer_directory}")
    for file in os.listdir(answer_directory):
        print("Loading option: ", file)
        img = Image.open(f"{answer_directory}/{file}").resize((114, 114))
        arr = np.array(img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        correct_answer_arrays.append(bgr)

    final_option_tiles = []

    for idx1, correct_answer in enumerate(correct_answer_arrays, start=1):
        best_score = -1
        best_match_to_answer = None
        for idx2, option in enumerate(options_data, start=1):
            print(f"Comparing correct answer {idx1} against game tile option {idx2}")
            option_bgr = option["np"]
            score = match_fp_option_tile(correct_answer, option_bgr)
            print(f"Score for correct answer {idx1} vs option {idx2}: {score:.4f}")
            if score > best_score:
                print(f"New best match for answer {idx1}: option {idx2} with score {score:.4f}")
                best_score = score
                best_match_to_answer = option
        print(f"Best match: {best_score:.4f}")
        if best_match_to_answer is not None:
            final_option_tiles.append(best_match_to_answer)
            options_data.remove(best_match_to_answer)

    testimg = Image.open(screenshot_filename).resize((1920, 1080))
    for final_option in final_option_tiles:
        draw = ImageDraw.Draw(testimg)
        x1, y1, x2, y2 = final_option["bbox"]
        draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline="red", width=5)
    testimg.show()



