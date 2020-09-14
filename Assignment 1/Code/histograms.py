from glob import glob
import os.path
import cv2 as cv
from matplotlib import pyplot

INPUT = [
    'data/card.png',
    'data/dugong.jpg',
    'results/card-scaled_*x.png',
    'results/card-rotated_*deg.png',
    'results/dugong-scaled_*x.png',
    'results/dugong-rotated_*deg.png',
]
OUTPUT_FILE = 'results/{}-hist.png'
HISTOGRAM_BINS = 16


for file in [f for p in INPUT for f in glob(p)]:
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    filename = os.path.splitext(os.path.basename(file))[0]

    pyplot.clf()
    pyplot.hist(image.flatten(), HISTOGRAM_BINS, (0, 255))
    pyplot.xlabel('Intensity')
    pyplot.ylabel('Frequency')
    pyplot.ylim((0, image.shape[0] * image.shape[1]))
    pyplot.tight_layout()
    output_file = OUTPUT_FILE.format(filename)
    pyplot.savefig(output_file)
