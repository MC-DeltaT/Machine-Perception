# Defines the SVM used to recognise digits.

import cv2 as cv
import numpy


__all__ = [
    'DigitRecognitionModel'
]


class DigitRecognitionModel:
    def __init__(self, file_path=None) -> None:
        if file_path is None:
            self.svm = cv.ml.SVM_create()
            self.svm.setKernel(cv.ml.SVM_LINEAR)
            self.svm.setType(cv.ml.SVM_C_SVC)
        else:
            self.svm = cv.ml.SVM_load(file_path)

    def train(self, inputs: numpy.ndarray, labels: numpy.ndarray) -> None:
        self.svm.trainAuto(inputs, cv.ml.ROW_SAMPLE, labels, 4)

    def predict_single(self, input: numpy.ndarray) -> int:
        return int(self.svm.predict(numpy.array([input]))[1][0][0])

    def predict_multiple(self, inputs: numpy.ndarray) -> numpy.ndarray:
        _, predictions = self.svm.predict(inputs)
        return predictions.ravel().astype(numpy.int32)

    def save(self, file_path) -> None:
        self.svm.save(str(file_path))
