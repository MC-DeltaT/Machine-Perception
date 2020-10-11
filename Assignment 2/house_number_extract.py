import cv2 as cv
from glob import glob
import numpy
import os.path
from region_detect import detect_regions

NUMBER_OUTPUT_FILE = 'output/HouseNumber{}.txt'
DETECTION_MODEL_FILE = 'output/detect_model.bin'
RECOGNITION_MODEL_FILE = 'output/recognition_model.bin'

detection_model = cv.ml.SVM_load(DETECTION_MODEL_FILE)
recognition_model = cv.ml.SVM_load(RECOGNITION_MODEL_FILE)
for input_file in glob('train/*') + glob('val/*'):
    filename = os.path.splitext(os.path.basename(input_file))[0]
    image = cv.imread(input_file, cv.IMREAD_COLOR)
    # Scale down large images for performance reasons.
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim > 500:
        scale = 500 / max_dim
        image = cv.resize(image, (0, 0), None, scale, scale)
    image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    boxes = detect_regions(image)
    result = image.copy()
    digits = []
    for x1, y1, x2, y2 in boxes:
        region = image_grey[y1:y2, x1:x2]
        _, region = cv.threshold(region, 0, 255, cv.THRESH_OTSU + cv.THRESH_TOZERO)
        region = cv.resize(region, (20, 20), interpolation=cv.INTER_CUBIC)
        region = cv.normalize(region, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        feature = region.ravel()
        is_digit = detection_model.predict(numpy.array([feature]))[1][0][0]
        if is_digit:
            result = cv.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
            digit = recognition_model.predict(numpy.array([feature]))[1][0][0]
            digits.append(int(digit))

    output_file = f'output/{filename}.png'
    cv.imwrite(output_file, result)

    output_file = NUMBER_OUTPUT_FILE.format(filename)
    with open(output_file, 'w') as file:
        file.write('Building number ' + ''.join(map(str, digits)))
