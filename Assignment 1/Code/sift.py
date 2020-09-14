from glob import glob
import os.path
import cv2 as cv

INPUT = [
    'data/card.png',
    'data/dugong.jpg',
    'results/card-scaled_*x.png',
    'results/card-rotated_*deg.png',
    'results/dugong-scaled_*x.png',
    'results/dugong-rotated_*deg.png',
]
HISTOGRAM_OUTPUT_FILE = 'results/{}-hist.png'
OUTPUT_FILE = 'results/{}-keypoints.png'
SCALE = 2

for file in [f for p in INPUT for f in glob(p)]:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]

    sift = cv.xfeatures2d.SIFT_create()
    keypoints = sift.detect(image, None)
    # Scale everything up so we can see it better (doesn't affect keypoint location)
    for kp in keypoints:
        kp.pt = (kp.pt[0] * SCALE, kp.pt[1] * SCALE)
        kp.size *= SCALE
    result = cv.resize(image, (0, 0), fx=SCALE, fy=SCALE)
    result = cv.drawKeypoints(result, keypoints, None)
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
