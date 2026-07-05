import numpy as np
import cv2
from scipy import ndimage


def extract_lane_points(mask, min_pixels=50):
    """
    Separates the binary mask into distinct lane blobs and returns
    (x, y) pixel coordinates for each lane.
    """
    labeled_mask, num_features = ndimage.label(mask)
    lanes = []

    for label_id in range(1, num_features + 1):
        ys, xs = np.where(labeled_mask == label_id)
        if len(xs) >= min_pixels:
            lanes.append((xs, ys))

    return lanes


def fit_polynomial(xs, ys, degree=2):
    """
    Fits x = f(y) since lanes are more vertical than horizontal in image space.
    Returns polynomial coefficients.
    """
    coeffs = np.polyfit(ys, xs, degree)
    return coeffs


def draw_lane_curves(image, mask, degree=2, min_y_span=30):
    """
    Fits curves to each lane in the mask and draws them on the image.
    Skips lanes with too little vertical span (likely noise) and
    limits curve drawing to the actual detected y-range (avoids
    wild extrapolation near the horizon).
    """
    image = np.array(image).copy()
    lanes = extract_lane_points(mask)
    fitted_curves = []

    for xs, ys in lanes:
        if len(ys) < 10:
            continue

        y_min, y_max = ys.min(), ys.max()
        if (y_max - y_min) < min_y_span:
            continue  

        coeffs = fit_polynomial(xs, ys, degree)
        fitted_curves.append(coeffs)

        
        y_range = np.linspace(y_min, y_max, int(y_max - y_min))
        x_fit = np.polyval(coeffs, y_range)

        points = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_range)
                            if 0 <= x < image.shape[1]], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=4)

    return image, fitted_curves


def compute_lane_offset(fitted_curves, image_width, image_height):
    """
    Estimates vehicle offset from lane center, in pixels.
    Assumes the two innermost curves (closest to image center) are the current lane boundaries.
    """
    if len(fitted_curves) < 2:
        return None

    bottom_y = image_height - 1
    x_positions = [np.polyval(c, bottom_y) for c in fitted_curves]
    x_positions.sort()

    # Find the two lane lines straddling the image center (assume current lane)
    image_center = image_width / 2
    left_candidates = [x for x in x_positions if x < image_center]
    right_candidates = [x for x in x_positions if x >= image_center]

    if not left_candidates or not right_candidates:
        return None

    left_lane_x = max(left_candidates)
    right_lane_x = min(right_candidates)

    lane_center = (left_lane_x + right_lane_x) / 2
    offset_pixels = image_center - lane_center

    return offset_pixels
