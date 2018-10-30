# encoding: utf-8

from Methods import OrientationField, Normalization, RidgeFrequency

def orientation_field(image, inputBlockSize):
    print "orientation_field"
    # image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\FingerPrint\\images\\fingerprint.jpg"
    print "main"
    return OrientationField.principal(image, inputBlockSize)

def normalization(inputMean, inputVariance):
    print "normalization"
    # image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\FingerPrint\\images\\fingerprint.jpg"
    print "main"
    return Normalization.principal(image, inputMean, inputVariance)

def ridge_frequency_extraction(inputBlockSize):
    image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\FingerPrint\\images\\fingerprint.jpg"
    print "main"
    return RidgeFrequency.principal(image, inputBlockSize)