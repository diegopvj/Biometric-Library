# encoding: utf-8

import Utils
import SmoothOrientationField

def main(image, inputBlockSize):
    originalImage = Utils.open_image(image)
    imageSize = Utils.get_size(originalImage)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    ridgeFrequencyExtracted = ridge_frequency_impl(imageConverted, imageSize, inputBlockSize)
    Utils.show_image(ridgeFrequencyExtracted)
    
    return ridgeFrequencyExtracted

def ridge_frequency_impl(imageConverted, imageSize, inputBlockSize):
    blockSize = int(inputBlockSize)
    orientationInEachBlock = SmoothOrientationField.gradient_orientation_smooth_impl(imageConverted, imageSize, blockSize)
    ridgeFrequency = Utils.ridge_frequency(imageConverted, blockSize, orientationInEachBlock)
    
    return ridgeFrequency 