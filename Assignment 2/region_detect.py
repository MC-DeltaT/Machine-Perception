import cv2 as cv
import numpy

REGION_PADDING = 0.1

def detect_regions(image):
    image = cv.medianBlur(image, 3)
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grey = cv.normalize(grey, None, 0, 255, cv.NORM_MINMAX)
    _, binary = cv.threshold(grey, 0, 255, cv.THRESH_OTSU)
    binary = cv.medianBlur(binary, 3)
    region_count, labels = cv.connectedComponents(binary)
    # Group points by region.
    region_points = [numpy.flip(numpy.argwhere(labels == l), 1) for l in range(region_count)]
    # Calculate region bounding boxes.
    boxes = [cv.boundingRect(ps) for ps in region_points]
    boxes = [b for b in boxes if region_filter(b)]
    adjusted_boxes = [
        (max(0, int(x - w * REGION_PADDING)),
         max(0, int(y - h * REGION_PADDING)),
         min(image.shape[1], int(x + w * (1 + REGION_PADDING))),
         min(image.shape[0], int(y + h * (1 + REGION_PADDING))))
        for x, y, w, h in boxes
    ]
    return adjusted_boxes

def region_filter(box):
    x, y, width, height = box

    # Filter out tiny regions, most likely noise and not digits.
    area = width * height
    if area < 100:
        return False

    # Digits typically have aspect ratios < 1, but not too low.
    aspect_ratio = width / height
    if not 0.2 < aspect_ratio < 0.8:
        return False

    return True
