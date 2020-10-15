# Trains the SVM used to recognise digits.

import cv2 as cv
import numpy


__all__ = [
    'DigitRecognitionModel'
]


class DigitRecognitionModel:
    def __init__(self, file=None) -> None:
        if file is None:
            self.svm = cv.ml.SVM_create()
            self.svm.setKernel(cv.ml.SVM_LINEAR)
            self.svm.setType(cv.ml.SVM_C_SVC)
        else:
            self.svm = cv.ml.SVM_load(file)

    def train(self, inputs: numpy.ndarray, labels: numpy.ndarray) -> None:
        self.svm.trainAuto(inputs, cv.ml.ROW_SAMPLE, labels)

    def predict(self, input: numpy.ndarray) -> int:
        return int(self.svm.predict(numpy.array([input]))[1][0][0])
