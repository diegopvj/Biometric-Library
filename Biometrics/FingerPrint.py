# encoding: utf-8
from Methods import OrientationField, Normalization, RidgeFrequency, Segmentation

def orientation_field(inputBlockSize):
    print "orientation_field"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\fingerprint.jpg"
    
    return OrientationField.principal(image, inputBlockSize)

def normalization(inputMean, inputVariance):
    print "normalization"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\fingerprint.jpg"
    
    return Normalization.principal(image, inputMean, inputVariance)

def ridge_frequency_extraction(inputBlockSize):
    print "ridge_frequency_extraction"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\fingerprint.jpg"
    
    return RidgeFrequency.principal(image, inputBlockSize)

def segmentation(inputBlockSize, inputTreshold):
    print "segmentation"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\fingerprint.jpg"
    
    return Segmentation.principal(image, inputBlockSize, inputTreshold)
