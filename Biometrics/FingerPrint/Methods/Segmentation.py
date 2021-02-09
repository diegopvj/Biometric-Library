# encoding: utf-8

from . import Utils
import itertools

def main(image, inputBlockSize, threshold):
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    segmentedImage, varianceImage = segmentation_and_variance_impl(imageConverted, inputBlockSize, threshold)
    Utils.show_image(segmentedImage)
    Utils.show_image(varianceImage)
    return segmentedImage, varianceImage


def segmentation_and_variance_impl(image, inputBlockSize, threshold):
    imageSize = Utils.get_size(image)
    baseSegmentedImage = Utils.image_copy(image)
    baseVarianceImage = Utils.image_copy(image)
    segmentedImage, varianceImage = Utils.segmentation(image, imageSize, inputBlockSize, threshold, baseSegmentedImage, baseVarianceImage)
    return (segmentedImage, varianceImage)