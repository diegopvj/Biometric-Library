# encoding: utf-8
import utils

sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

def principal(image, inputBlockSize):
    print "depois"
    originalImage = utils.open_image(image)
    blockSize = int(inputBlockSize)
    imageConverted = utils.convert_to_black_and_white(originalImage)
    showImage = utils.show_image(imageConverted)
    gradientOrientation = gradient_orientation_impl(imageConverted, blockSize)
    return gradientOrientation

def gradient_orientation_impl(image, blockSize):
    size = utils.get_size(image)
    coordinate = utils.get_coordinate(size)
    getPixel = utils.get_pixel(image, (coordinate['x'], coordinate['y']))
    sobelCoordinate = utils.get_sobel_coordinates(sobelOperator)

    result = [[] for i in range(1, coordinate['x'], blockSize)]

    for i in range(1, coordinate['x'], blockSize):
        for j in range(1, coordinate['y'], blockSize):
            print('i,j', (i,j))
            # print('i', i)
            # print('j', j)
    return result