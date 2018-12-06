# encoding: utf-8

from Methods import OrientationField, Normalization, RidgeFrequency

def orientation_field(inputBlockSize):
    print "orientation_field"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\105_5.tif"
    # image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\FingerPrint\\images\\fingerprint.jpg"
    return OrientationField.principal(image, inputBlockSize)

def normalization(inputMean, inputVariance):
    print "normalization"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\107_7.tif"
    # image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\FingerPrint\\images\\fingerprint.jpg"
    return Normalization.principal(image, inputMean, inputVariance)

def ridge_frequency_extraction(inputBlockSize):
    print "ridge_frequency_extraction"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\107_7.tif"
    # image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\FingerPrint\\images\\fingerprint.jpg"
    return RidgeFrequency.principal(image, inputBlockSize)