# encoding: utf-8

from . import Utils
from . import SmoothOrientationField
from . import RidgeFrequency

def main(image, inputBlockSize, xSigma, ySigma):
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    imageSize = Utils.get_size(imageConverted)
    imageFiltered = gabor_filter_impl(imageConverted, imageSize, inputBlockSize, xSigma, ySigma)
    Utils.show_image(imageFiltered)

    return imageFiltered

def gabor_filter_impl(imageConverted, imageSize, inputBlockSize, xSigma, ySigma):
    blockSize = int(inputBlockSize)
    orientationInEachBlock = SmoothOrientationField.gradient_orientation_smooth_impl(imageConverted, imageSize, inputBlockSize)
    print("local ridge orientation")
    imageLoaded = Utils.image_load(imageConverted)
    frequencys = Utils.image_frequencys(imageSize, imageLoaded, blockSize, orientationInEachBlock)
    print("local ridge frequency")
    gaborApplied = Utils.gabor_filter(imageConverted, imageLoaded, imageSize, blockSize, xSigma, ySigma, orientationInEachBlock, frequencys)
    
    return gaborApplied