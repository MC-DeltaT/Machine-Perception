import cv2 as cv
import numpy

DIGITS_FILE = 'training_digits.png'
MODEL_OUTPUT_FILE = 'classification_model.bin'
DIGIT_SIZE = 20
DIGITS = 10
DIGIT_ROWS = 5
SVM_C = 1
ITERATIONS = 100
EPSILON = 0.001

digits = cv.imread(DIGITS_FILE, cv.IMREAD_GRAYSCALE)
rows = [digits[i * DIGIT_SIZE: (i + 1) * DIGIT_SIZE] for i in range(DIGITS * DIGIT_ROWS)]
rows = [r.T.reshape((-1, DIGIT_SIZE, DIGIT_SIZE)) for r in rows]
digits = numpy.array([numpy.concatenate(rows[i * DIGIT_ROWS:(i + 1) * DIGIT_ROWS], 0) for i in range(DIGITS)])
labels = numpy.array([numpy.full((d.shape[0]), l, numpy.int) for l, d in enumerate(digits)])

features = digits.reshape((digits.shape[0], digits.shape[1], DIGIT_SIZE * DIGIT_SIZE)).astype(numpy.float32)
mins = numpy.min(features, 2)
maxs = numpy.max(features, 2)
features = (features - mins[:, :, numpy.newaxis]) / (maxs - mins)[:, :, numpy.newaxis]

training_inputs, test_inputs = numpy.split(features, 2, 1)
training_inputs = training_inputs.reshape((training_inputs.shape[0] * training_inputs.shape[1], -1))
test_inputs = test_inputs.reshape((test_inputs.shape[0] * test_inputs.shape[1], -1))
training_labels, test_labels = numpy.split(labels, 2, 1)
training_labels = training_labels.ravel()
test_labels = test_labels.ravel()

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_RBF)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(SVM_C)
svm.setGamma(0.1)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, ITERATIONS, EPSILON))
svm.train(training_inputs, cv.ml.ROW_SAMPLE, training_labels)
svm.save(MODEL_OUTPUT_FILE)

for name, inputs, labels in [('training', training_inputs, training_labels), ('test', test_inputs, test_labels)]:
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
