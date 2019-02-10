# encoding: utf-8

import Utils
import OrientationField

def principal(image, inputBlockSize, gaussKernelSize, gaussSigma):
    blockSize = int(inputBlockSize)
    originalImage = Utils.open_image(image)
    imageSize = Utils.get_size(originalImage)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    gradientOrientationSmoothed = gradient_orientation_smooth(imageConverted, imageSize, inputBlockSize, gaussKernelSize, gaussSigma)
    drawSmoothedOrientationLines = Utils.make_lines(imageConverted, imageSize, gradientOrientationSmoothed, blockSize)
    Utils.show_image(drawSmoothedOrientationLines)
    
    return drawSmoothedOrientationLines

def gradient_orientation_smooth(imageConverted, imageSize, inputBlockSize, gaussKernelSize, gaussSigma):
    gradientOrientations = OrientationField.gradient_orientation_extraction_impl(imageConverted, imageSize, inputBlockSize)
    smoothedOrientations = Utils.smooth_orientations(gradientOrientations, gaussKernelSize, gaussSigma)

    return smoothedOrientations