# encoding: utf-8

from . import Utils
from . import OrientationField

def main(image, inputBlockSize):
    blockSize = int(inputBlockSize)
    originalImage = Utils.open_image(image)
    imageSize = Utils.get_size(originalImage)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    gradientOrientationSmoothed = gradient_orientation_smooth_impl(imageConverted, imageSize, inputBlockSize)
    drawSmoothedOrientationLines = Utils.make_lines(imageConverted, imageSize, gradientOrientationSmoothed, blockSize)
    Utils.show_image(drawSmoothedOrientationLines)
    
    return drawSmoothedOrientationLines

def gradient_orientation_smooth_impl(imageConverted, imageSize, inputBlockSize):
    gradientOrientations = OrientationField.gradient_orientation_extraction_impl(imageConverted, imageSize, inputBlockSize)
    smoothedOrientations = Utils.smooth_orientations(gradientOrientations)

    return smoothedOrientations