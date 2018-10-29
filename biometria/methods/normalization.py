# encoding: utf-8
import utils

def principal(im, inputMean, inputVariance):
    print "principal"
    image = utils.open_image(im)
    imageConverted = utils.convert_to_black_and_white(image)
    imageConverted.show()
    normalization = normalize(imageConverted, inputMean, inputVariance)
    normalization.show()

def normalize(image, inputMean, inputVariance):
    float(inputMean)
    float(inputVariance)
    mean = utils.image_mean(image)
    standardDeviation = utils.image_standard_deviation(image)
    variance = standardDeviation ** 2
    return image.point(lambda x: utils.normalize_pixel(x, inputVariance, variance, inputMean, mean))