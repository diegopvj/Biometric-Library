# encoding: utf-8
import Utils

sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

def principal(image, inputBlockSize):
    print "depois"
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    showImage = Utils.show_image(imageConverted)
    gradientOrientation = gradient_orientation_impl(imageConverted, inputBlockSize)
    return gradientOrientation

def gradient_orientation_extraction_impl(image, inputBlockSize):
    blockSize = int(inputBlockSize)
    size = Utils.get_size(image)
    coordinate = Utils.get_coordinate(size)
    getPixel = Utils.get_pixel(image, (coordinate['x'], coordinate['y']))
    sobelCoordinate = Utils.get_sobel_coordinates(sobelOperator)

    result = [[] for i in range(1, coordinate['x'], blockSize)]

    for i in range(1, coordinate['x'], blockSize):
        for j in range(1, coordinate['y'], blockSize):
            print('i,j', (i,j))
            # print('i', i)
            # print('j', j)
    return result