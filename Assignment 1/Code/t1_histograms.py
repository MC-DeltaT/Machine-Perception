# Plots histograms of the scaled and rotated images.
# gen_scaled.py and gen_rotated.py should be run before running this script.

from glob import glob
import os.path
from pathlib import Path
import cv2 as cv
from matplotlib import pyplot

BASE_IMAGES = ['card', 'dugong']
TRANSFORMS = ['scaled', 'rotated']
INPUT_FILES = 'generated/{}/{}/*.png'
OUTPUT_DIR = 'results/{}/{}'
OUTPUT_FILE = '{}-hist.png'
HISTOGRAM_BINS = 16

for base_image in BASE_IMAGES:
    for transform in TRANSFORMS:
        file_pattern = INPUT_FILES.format(base_image, transform)
        output_dir = OUTPUT_DIR.format(base_image, transform)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for file in glob(file_pattern):
            image = cv.imread(file, cv.IMREAD_GRAYSCALE)
            filename = os.path.splitext(os.path.basename(file))[0]

            pyplot.clf()
            pyplot.hist(image.flatten(), HISTOGRAM_BINS, (0, 255))
            pyplot.xlabel('Intensity')
            pyplot.ylabel('Frequency')
            pyplot.ylim((0, image.shape[0] * image.shape[1]))
            pyplot.tight_layout()
            output_file = os.path.join(output_dir, OUTPUT_FILE.format(filename))
            pyplot.savefig(output_file)
