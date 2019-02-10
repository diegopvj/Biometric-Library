# encoding: utf-8

import Utils
import json

sobelMask = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

def principal(image, inputBlockSize):
    blockSize = int(inputBlockSize)
    originalImage = Utils.open_image(image)
    imageSize = Utils.get_size(originalImage)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    gradientOrientation = gradient_orientation_extraction_impl(imageConverted, imageSize, blockSize)
    drawOrientationLines = Utils.make_lines(imageConverted, imageSize, gradientOrientation, blockSize)
    Utils.show_image(drawOrientationLines)
    
    return drawOrientationLines

def gradient_orientation_extraction_impl(imageConverted, imageSize, blockSize):
    setPixel = Utils.set_pixel(imageConverted, (imageSize['x'], imageSize['y']))
    sobelCoordinate = get_sobel_coordinates(sobelMask)
    orientationInEachBlock = Utils.calculate_orientation_in_each_block(blockSize, imageSize, sobelCoordinate, setPixel)
    
    return orientationInEachBlock

def get_sobel_coordinates(sobelMask):
    sobel = json.dumps({'x': Utils.transpose(sobelMask), 'y': sobelMask})
    
    return json.loads(sobel)
