import cv2 as cv
import numpy
from pathlib import Path
from random import randrange


INPUTS_FILE = 'generated/digit_recognition_training_inputs.npy'
LABELS_FILE = 'generated/digit_recognition_training_labels.npy'
MODEL_OUTPUT_FILE = Path('generated/recognition_model.xml')


inputs = numpy.load(INPUTS_FILE)
labels = numpy.load(LABELS_FILE)

# random_seed = randrange(2 ** 32 - 1)
# numpy.random.default_rng(random_seed).shuffle(inputs)
# numpy.random.default_rng(random_seed).shuffle(labels)
#
# training_inputs, test_inputs = numpy.split(inputs, 2)
# training_labels, test_labels = numpy.split(labels, 2)
#
# print(f'Training dataset size: {training_inputs.shape[0]}')
# print(f'Test dataset size: {test_inputs.shape[0]}')
# print()

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.trainAuto(inputs, cv.ml.ROW_SAMPLE, labels)
MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
svm.save(str(MODEL_OUTPUT_FILE))

for name, inputs, labels in [('training', inputs, labels)]:
    predictions = svm.predict(inputs)[1]
    predictions = predictions.ravel()
    accuracy = numpy.sum(predictions == labels) / inputs.shape[0]
    print(f'{name}: {round(accuracy * 100, 2)}% accuracy')

    confusion_matrix = numpy.empty((10, 10), numpy.uint16)
    for i in range(10):
        label_mask = labels == i
        for j in range(10):
            prediction_mask = predictions == j
            confusion_matrix[i, j] = numpy.sum(numpy.logical_and(label_mask, prediction_mask))

    cell_width = max(len(str(e)) for e in confusion_matrix.ravel())
    cell_header = '  | ' + ' '.join(str(e).rjust(cell_width) for e in range(10))
    print(cell_header)
    print('-' * len(cell_header))
    for i, row in enumerate(confusion_matrix):
        row_str = ' '.join((str(e).rjust(cell_width) for e in row))
        print(f'{i} | {row_str}')

    print()
