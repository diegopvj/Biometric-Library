# encoding: utf-8

import Utils
import SmoothOrientationField
import RidgeFrequency

def principal(image, inputBlockSize, xSigma, ySigma):
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    imageSize = Utils.get_size(imageConverted)
    imageFiltered = gabor_filter_impl(imageConverted, imageSize, inputBlockSize, xSigma, ySigma).show()
    
    return imageFiltered

def gabor_filter_impl(imageConverted, imageSize, inputBlockSize, xSigma, ySigma):
    blockSize = int(inputBlockSize)
    orientationInEachBlock = SmoothOrientationField.gradient_orientation_smooth(imageConverted, imageSize, inputBlockSize)
    print "local ridge orientation"
    imageLoad = Utils.image_load(imageConverted)
    frequencys = Utils.image_frequencys(imageSize, imageLoad, blockSize, orientationInEachBlock)
    print "local ridge frequency"
    gaborApplied = Utils.gabor_filter(imageConverted, imageLoad, imageSize, blockSize, xSigma, ySigma, orientationInEachBlock, frequencys)
    
    return gaborApplied