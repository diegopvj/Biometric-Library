# encoding: utf-8

import Utils
import OrientationField

def principal(image, inputBlockSize):
    print "depois"
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    showImage = Utils.show_image(imageConverted)
    ridgeFrequencyExtracted = ridge_frequency_impl(imageConverted, inputBlockSize)
    return ridgeFrequencyExtracted

def ridge_frequency_impl(image, inputBlockSize):
    blockSize = int(inputBlockSize)
    orientationInEachBlock = OrientationField.gradient_orientation_extraction_impl(image, blockSize)
    ridgeFrequency = Utils.ridge_frequency(image, blockSize, orientationInEachBlock)
    return ridgeFrequency