import cv2 as cv
import numpy
from pathlib import Path
from sys import argv


if len(argv) != 4:
    print('Usage: python3 recognition_model.py <inputs_file> <labels_file> <output_file>')
    exit(1)

inputs_file = argv[1]
labels_file = argv[2]
output_file = Path(argv[3])

inputs = numpy.load(inputs_file)
labels = numpy.load(labels_file)

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.trainAuto(inputs, cv.ml.ROW_SAMPLE, labels)
output_file.parent.mkdir(parents=True, exist_ok=True)
svm.save(str(output_file))
