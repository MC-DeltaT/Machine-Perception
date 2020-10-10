import cv2 as cv
from glob import glob
import numpy
import os.path


def region_filter(image, box, points):
    x, y, width, height = box

    # Filter out tiny regions, most likely noise and not digits.
    area = width * height
    if area < image.shape[0] / 20 * image.shape[1] / 20:
        return False

    # Areas with extreme foreground ratio are likely noise.
    foreground_ratio = points.shape[0] / area
    if not 0.1 < foreground_ratio < 0.9:
        return False

    # Digits typically have aspect ratios < 1, but not too low.
    aspect_ratio = width / height
    if not 0.2 < aspect_ratio < 0.8:
        return False

    # All the digits seem to be high intensity on dark background. Therefore there should
    # be 2, compact, well-separated clusters of pixel values.
    # roi = image[y:y+height, x:x+width]
    # roi = cv.normalize(roi, None, 0, 255, cv.NORM_MINMAX)
    # roi = roi / 255
    # kmeans_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    # kmeans_flags = cv.KMEANS_RANDOM_CENTERS
    # roi_feature = roi.astype(numpy.float32).reshape(-1, 3)
    # compactness, _, kmeans_centres = cv.kmeans(roi_feature, 2, None, kmeans_criteria, 5, kmeans_flags)
    # c1, c2 = kmeans_centres
    # print(numpy.linalg.norm(c1 - c2), compactness / roi_feature.shape[0])
    # if numpy.linalg.norm(c1 - c2) < 1:
    #     return False

    return True


for input_file in [*glob('train/*.*'), *glob('val/*.*')]:
    image = cv.imread(input_file, cv.IMREAD_COLOR)
    image = cv.medianBlur(image, 3)
    # Scale down large images for performance reasons.
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim > 500:
        scale = 500 / max_dim
        image = cv.resize(image, (0, 0), None, scale, scale)
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Normalise for more reliable thresholding.
    grey = cv.normalize(grey, None, 0, 255, cv.NORM_MINMAX)

    # Use watershed to find regions. Performs better than just binarisation when there is variation
    # in lighting across single regions.
    # TODO: better marker generation
    background = grey < 50
    foreground = grey > 150
    unknown = numpy.logical_not(numpy.logical_or(background, foreground))
    _, fg_regions = cv.connectedComponents(foreground.astype(numpy.uint8))
    markers = fg_regions + 1
    markers[unknown] = 0
    markers = markers.astype(numpy.int32)
    labels = cv.watershed(image, markers)
    region_count = labels.max() - labels.min() + 1

    # Group points by region.
    region_points = numpy.array([numpy.flip(numpy.argwhere(labels == l), 1) for l in range(region_count)])
    # Calculate region bounding boxes.
    boxes = numpy.array([cv.boundingRect(ps) for ps in region_points])

    result = image.copy()
    for box, points in zip(boxes, region_points):
        if region_filter(image, box, points):
            x1, y1, width, height = box
            x2 = x1 + width
            y2 = y1 + height
            cv.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f'output/{filename}.png'
    cv.imwrite(output_file, result)
