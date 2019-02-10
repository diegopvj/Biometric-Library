# encoding: utf-8

import Utils
import SmoothOrientationField

def principal(image, inputBlockSize, gaussKernelSize, gaussSigma):
    originalImage = Utils.open_image(image)
    imageSize = Utils.get_size(originalImage)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    ridgeFrequencyExtracted = ridge_frequency_impl(imageConverted, imageSize, inputBlockSize, gaussKernelSize, gaussSigma)
    Utils.show_image(ridgeFrequencyExtracted)
    return ridgeFrequencyExtracted

def ridge_frequency_impl(imageConverted, imageSize, inputBlockSize, gaussKernelSize, gaussSigma):
    blockSize = int(inputBlockSize)
    orientationInEachBlock = SmoothOrientationField.gradient_orientation_smooth(imageConverted, imageSize, blockSize, gaussKernelSize, gaussSigma)
    ridgeFrequency = Utils.ridge_frequency(imageConverted, blockSize, orientationInEachBlock)
    return ridgeFrequency
    

    imageConverted, imageSize, inputBlockSize, gaussKernelSize, gaussSigma