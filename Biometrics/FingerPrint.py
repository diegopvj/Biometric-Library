# encoding: utf-8

from Methods import OrientationField, SmoothOrientationField, Normalization, RidgeFrequency, Segmentation, GaborFilter

def orientation_field_extraction(inputBlockSize):
    print "orientation_field"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return OrientationField.principal(image, inputBlockSize)

def smooth_orientation_field(inputBlockSize):
    print "smooth_orientation_field"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return SmoothOrientationField.principal(image, inputBlockSize)

def normalization(inputMean, inputVariance):
    print "normalization"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    return Normalization.principal(image, inputMean, inputVariance)

def ridge_frequency_extraction(inputBlockSize):
    print "ridge_frequency_extraction"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return RidgeFrequency.principal(image, inputBlockSize)

def segmentation(inputBlockSize, inputTreshold):
    print "segmentation"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return Segmentation.principal(image, inputBlockSize, inputTreshold)

def gabor_filter(inputBlockSize, xSigma, ySigma):
    print "gabor_filter"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return GaborFilter.principal(image, inputBlockSize, xSigma, ySigma)
