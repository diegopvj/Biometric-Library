# encoding: utf-8

import Utils
import itertools

def principal(image, inputBlockSize, threshold):
    print "depois"
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    showImage = Utils.show_image(imageConverted)
    segmentedImage, varianceImage = segmented_and_variance_impl(imageConverted, inputBlockSize, threshold)
    segmentedImage.show()
    varianceImage.show()
    return segmentedImage, varianceImage


def segmented_and_variance_impl(image, inputBlockSize, threshold):
    imageSize = Utils.get_size(image)
    baseSegmentedImage = Utils.image_copy(image)
    baseVarianceImage = Utils.image_copy(image)
    segmentedImage, varianceImage = Utils.segmentation(image, imageSize, inputBlockSize, threshold, baseSegmentedImage, baseVarianceImage)
    return (segmentedImage, varianceImage)