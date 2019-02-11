# encoding: utf-8

import Utils

def principal(image):
    originalImage = Utils.open_image(image)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    thinnedImage = thinning_impl(imageConverted).show()
    return thinnedImage

def thinning_impl(imageConverted):
    imageSize = Utils.get_size(imageConverted)
    pixelsInImageLoaded = Utils.get_pixels_in_image_loaded(imageConverted, imageSize)
    
    return Utils.thinning_structures(imageConverted, imageSize, pixelsInImageLoaded)
