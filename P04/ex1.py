from glob import glob
import os.path

import cv2 as cv


INPUT_FILES = 'data/prac04ex01img*'
KEYPOINT_FILE = 'output/{}_keypoints.png'
DESCRIPTOR_FILE = 'output/{}_descriptors.png'

for file in glob(INPUT_FILES):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    sift = cv.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    filename = os.path.splitext(os.path.basename(file))[0]
    output = cv.drawKeypoints(image, keypoints, None)
    output_file = KEYPOINT_FILE.format(filename)
    cv.imwrite(output_file, output)

    output_file = DESCRIPTOR_FILE.format(filename)
    output = cv.normalize(descriptors, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    cv.imwrite(output_file, output)

