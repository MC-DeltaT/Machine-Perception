import cv2 as cv
from digit_descriptor import digit_descriptor
import numpy
from os import listdir
import os.path
from pathlib import Path


SAMPLES_OUTPUT_FILE = Path('generated/digit_recognition_training_inputs.npy')
LABELS_OUTPUT_FILE = Path('generated/digit_recognition_training_labels.npy')

inputs = []
labels = []


# Handwritten digit collation from previous practical.

# HANDWRITTEN_DIGIT_FILE = 'data/digits/handwritten.png'
# HANDWRITTEN_DIGIT_SIZE = 20
# HANDWRITTEN_DIGIT_ROWS = 5
#
# image = cv.imread(HANDWRITTEN_DIGIT_FILE, cv.IMREAD_GRAYSCALE)
# samples_per_row = image.shape[1] // HANDWRITTEN_DIGIT_SIZE
# rows = [image[i * HANDWRITTEN_DIGIT_SIZE: (i + 1) * HANDWRITTEN_DIGIT_SIZE]
#         for i in range(HANDWRITTEN_DIGIT_ROWS * 10)]
# rows = [numpy.array(numpy.hsplit(r, samples_per_row)) for r in rows]
# digits = numpy.vstack(rows)
# descriptors = numpy.array([preprocess_digit(d) for d in digits])
# inputs.extend(descriptors)
# samples_per_digit = HANDWRITTEN_DIGIT_ROWS * image.shape[1] // HANDWRITTEN_DIGIT_SIZE
# for i in range(10):
#     labels.extend(numpy.full(samples_per_digit, i))


# Supplied individual digits.

INPUT_DIGIT_DIR = 'data/digits/{}'

for i in range(10):
    directory = INPUT_DIGIT_DIR.format(i)
    for entry in listdir(directory):
        file_path = os.path.join(directory, entry)
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        descriptor = digit_descriptor(image)
        inputs.append(descriptor)
        labels.append(i)


inputs = numpy.array(inputs, numpy.float32)
labels = numpy.array(labels)
SAMPLES_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LABELS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
numpy.save(SAMPLES_OUTPUT_FILE, inputs)
numpy.save(LABELS_OUTPUT_FILE, labels)
