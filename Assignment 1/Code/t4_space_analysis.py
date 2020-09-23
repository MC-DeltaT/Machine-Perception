# Visualises the feature spaces used for task 4 segmentation.

import cv2 as cv
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os.path
from pathlib import Path

INPUTS = ['data/card.png', 'data/dugong.jpg']
OUTPUT_DIR = 'results/{}/segmentation'
OUTPUT_FILE = '{}-space.png'
SPACE_SAMPLE_STRIDE = 4

def visualiser_1d(axis1):
    def inner(data, output_file):
        pyplot.clf()
        pyplot.hist(data.ravel(), 100)
        pyplot.xlabel(axis1)
        pyplot.ylabel('Frequency')
        pyplot.tight_layout()
        pyplot.savefig(output_file)
    return inner

# def visualiser_2d(axis1, axis2):
#     def inner(data, output_file):
#         pyplot.clf()
#         pyplot.scatter(data[:, 0], data[:, 1])
#         pyplot.xlabel(axis1)
#         pyplot.ylabel(axis2)
#         pyplot.tight_layout()
#         pyplot.savefig(output_file)
#     return inner

def visualiser_3d(axis1, axis2, axis3):
    def inner(data, output_file):
        pyplot.clf()
        figure = pyplot.figure()
        axes = figure.add_subplot(111, projection='3d')
        axes.scatter(data[:, 0].ravel(), data[:, 1].ravel(), data[:, 2].ravel())
        axes.set_xlabel(axis1)
        axes.set_ylabel(axis2)
        axes.set_zlabel(axis3)
        pyplot.tight_layout()
        pyplot.savefig(output_file)
    return inner

for file in INPUTS:
    image = cv.imread(file, cv.IMREAD_COLOR)
    filename = os.path.splitext(os.path.basename(file))[0]
    output_dir = OUTPUT_DIR.format(filename)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    features = [
        ('intensity', cv.cvtColor(image, cv.COLOR_BGR2GRAY)[:, :, numpy.newaxis], visualiser_1d('Intensity')),
        ('rgb', image, visualiser_3d('Blue', 'Green', 'Red')),
        ('hsv', cv.cvtColor(image, cv.COLOR_BGR2HSV), visualiser_3d('Hue', 'Saturation', 'Value')),
        ('hls', cv.cvtColor(image, cv.COLOR_BGR2HLS), visualiser_3d('Hue', 'Lightness', 'Saturation')),
        ('xyz', cv.cvtColor(image, cv.COLOR_BGR2XYZ), visualiser_3d('X', 'Y', 'Z')),
        ('lab', cv.cvtColor(image, cv.COLOR_BGR2LAB), visualiser_3d('L', 'a', 'b')),
        ('luv', cv.cvtColor(image, cv.COLOR_BGR2LUV), visualiser_3d('L', 'u', 'v')),
        ('yuv', cv.cvtColor(image, cv.COLOR_BGR2YUV), visualiser_3d('Y', 'U', 'V')),
        ('red', image[:, :, 2][:, :, numpy.newaxis], visualiser_1d('Red')),
        ('green', image[:, :, 1][:, :, numpy.newaxis], visualiser_1d('Green')),
        ('blue', image[:, :, 0][:, :, numpy.newaxis], visualiser_1d('Blue')),
        ('hue', cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 0][:, :, numpy.newaxis], visualiser_1d('Hue')),
        ('saturation', cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 1][:, :, numpy.newaxis], visualiser_1d('Saturation'))
    ]

    for name, data, visualiser in features:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2])).astype(numpy.float32)
        data = (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
        data_samples = data[::SPACE_SAMPLE_STRIDE]
        output_file = os.path.join(output_dir, OUTPUT_FILE.format(name))
        visualiser(data_samples, output_file)
