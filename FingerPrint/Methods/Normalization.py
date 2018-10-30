# encoding: utf-8
import Utils

def principal(im, inputMean, inputVariance):
    print "principal"
    image = Utils.open_image(im)
    imageConverted = Utils.convert_to_black_and_white(image)
    imageConverted.show()
    normalization = normalize(imageConverted, inputMean, inputVariance)
    normalization.show()

def normalize(image, inputMean, inputVariance):
    float(inputMean)
    float(inputVariance)
    mean = Utils.image_mean(image)
    standardDeviation = Utils.image_standard_deviation(image)
    variance = standardDeviation ** 2
    return image.point(lambda x: Utils.normalize_pixel(x, inputVariance, variance, inputMean, mean))