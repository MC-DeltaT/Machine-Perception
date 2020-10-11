import cv2 as cv
from glob import glob
import numpy
from random import shuffle

MODEL_OUTPUT_FILE = 'output/detect_model.bin'
REGION_SIZE = 20

positives = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob('regions/positive/*')]
negatives = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob('regions/negative/*')]
inputs = numpy.array([positives + negatives])
inputs = inputs.reshape((-1, REGION_SIZE * REGION_SIZE)).astype(numpy.float32)
mins = numpy.min(inputs, 1)
maxs = numpy.max(inputs, 1)
inputs = (inputs - mins[:, numpy.newaxis]) / (maxs - mins)[:, numpy.newaxis]
labels = numpy.array([1] * len(positives) + [0] * len(negatives))

tmp = list(zip(inputs, labels))
shuffle(tmp)
inputs, labels = zip(*tmp)
inputs = numpy.array(inputs)
labels = numpy.array(labels)

training_inputs, test_inputs = numpy.array_split(inputs, 2)
training_labels, test_labels = numpy.array_split(labels, 2)

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_RBF)
svm.setType(cv.ml.SVM_C_SVC)
svm.trainAuto(training_inputs, cv.ml.ROW_SAMPLE, training_labels)
svm.save(MODEL_OUTPUT_FILE)

for name, inputs, labels in [('training', training_inputs, training_labels), ('test', test_inputs, test_labels)]:
    predictions = svm.predict(inputs)[1]
    predictions = predictions.ravel()
    accuracy = numpy.sum(predictions == labels) / inputs.shape[0]
    print(f'{name}: {round(accuracy * 100, 2)}% accuracy')

    confusion_matrix = numpy.empty((2, 2), numpy.uint16)
    for i in range(2):
        label_mask = labels == i
        for j in range(2):
            prediction_mask = predictions == j
            confusion_matrix[i, j] = numpy.sum(numpy.logical_and(label_mask, prediction_mask))

    cell_width = max(len(str(e)) for e in confusion_matrix.ravel())
    cell_header = '  | ' + ' '.join(str(e).rjust(cell_width) for e in range(2))
    print(cell_header)
    print('-' * len(cell_header))
    for i, row in enumerate(confusion_matrix):
        row_str = ' '.join((str(e).rjust(cell_width) for e in row))
        print(f'{i} | {row_str}')

    print()

