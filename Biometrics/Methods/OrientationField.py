# encoding: utf-8

import Utils
import json
import os

sobelMask = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

def principal(image, inputBlockSize):
    print "depois"
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    imageConverted.show()
    gradientOrientation = gradient_orientation_extraction_impl(imageConverted, inputBlockSize)
    return gradientOrientation

def gradient_orientation_extraction_impl(image, inputBlockSize):
    blockSize = int(inputBlockSize)
    imageSize = Utils.get_size(image)
    setPixel = Utils.set_pixel(image, (imageSize['x'], imageSize['y']))
    sobelCoordinate = get_sobel_coordinates(sobelMask)
    orientationInEachBlock = Utils.calculate_orientation_in_each_block(blockSize, imageSize, sobelCoordinate, setPixel)
    # return Utils.make_lines(image, imageSize, orientationInEachBlock, blockSize).show()

    return orientationInEachBlock

def get_sobel_coordinates(sobelMask):
    sobel = json.dumps({'x': Utils.transpose(sobelMask), 'y': sobelMask})
    return json.loads(sobel)