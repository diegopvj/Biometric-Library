# encoding: utf-8
import Utils

def principal(im, inputMean, inputVariance):
    image = Utils.open_image(im)
    imageConverted = Utils.convert_to_black_and_white(image)
    normalization = normalize(imageConverted, inputMean, inputVariance)
    Utils.show_image(normalization)

    return normalization

def normalize(image, inputMean, inputVariance):
    m0 = float(inputMean)
    v0 = float(inputVariance)
    mean = Utils.image_mean(image)
    standardDeviation = Utils.image_standard_deviation(image)
    variance = standardDeviation ** 2
    
    return image.point(lambda x: Utils.normalize_pixel(x, v0, variance, m0, mean))