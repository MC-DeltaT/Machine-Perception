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

for file in [f for p in INPUT for f in glob(p)]:
    image = cv.imread(file, cv.IMREAD_COLOR)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filename = os.path.splitext(os.path.basename(file))[0]

    sift = cv.xfeatures2d.SIFT_create()
    keypoints = sift.detect(image, None)
    result = cv.drawKeypoints(image.copy(), keypoints, None)
    output_file = OUTPUT_FILE.format(filename)
    cv.imwrite(output_file, result)
