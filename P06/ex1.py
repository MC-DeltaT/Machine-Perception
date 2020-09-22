import cv2 as cv
import numpy

def max_response(m):
    return numpy.unravel_index(numpy.argmax(m), m.shape)

def min_response(m):
    return numpy.unravel_index(numpy.argmin(m), m.shape)

IMAGE_FILE = 'data/prac06ex01_image.png'
TEMPLATE_FILE = 'data/prac06ex01_template.png'
MATCH_RESPONSE_OUTPUT_FILE = 'output/{}-response.png'
MATCH_OUTPUT_FILE = 'output/{}-match.png'
METHODS = [
    ('sqdiff', cv.TM_SQDIFF, min_response),
    ('sqdiff_norm', cv.TM_SQDIFF_NORMED, min_response),
    ('ccorr', cv.TM_CCORR, max_response),
    ('ccorr_norm', cv.TM_CCORR_NORMED, max_response),
    ('ccoeff', cv.TM_CCOEFF, max_response),
    ('ccoeff_norm', cv.TM_CCOEFF_NORMED, max_response)
]

image = cv.imread(IMAGE_FILE, cv.IMREAD_COLOR)
template = cv.imread(TEMPLATE_FILE, cv.IMREAD_COLOR)
template_width = template.shape[1]
template_height = template.shape[0]
for method_name, method, match_selector in METHODS:
    match_response = cv.matchTemplate(image, template, method)
    match_response = cv.normalize(match_response, None, 0, 1, cv.NORM_MINMAX)

    result = (match_response * 255).astype(numpy.uint8)
    output_file = MATCH_RESPONSE_OUTPUT_FILE.format(method_name)
    cv.imwrite(output_file, result)

    match_y, match_x = match_selector(match_response)
    tl = (match_x, match_y)
    br = (match_x + template_width, match_y + template_height)
    result = image.copy()
    result = cv.rectangle(result, tl, br, (0, 255, 0))
    output_file = MATCH_OUTPUT_FILE.format(method_name)
    cv.imwrite(output_file, result)
