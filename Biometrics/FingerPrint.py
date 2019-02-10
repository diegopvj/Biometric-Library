# encoding: utf-8

from Methods import OrientationField, SmoothOrientationField, Normalization, RidgeFrequency, Segmentation

def orientation_field_extraction(inputBlockSize):
    print "orientation_field"
    image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    # image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return OrientationField.principal(image, inputBlockSize)

def smooth_orientation_field(inputBlockSize, gaussKernelSize, gaussSigma):
    print "smooth_orientation_field"
    image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    # image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return SmoothOrientationField.principal(image, inputBlockSize, gaussKernelSize, gaussSigma)

def normalization(inputMean, inputVariance):
    print "normalization"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    return Normalization.principal(image, inputMean, inputVariance)

def ridge_frequency_extraction(inputBlockSize, gaussKernelSize, gaussSigma):
    print "ridge_frequency_extraction"
    image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    # image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return RidgeFrequency.principal(image, inputBlockSize, gaussKernelSize, gaussSigma)

def segmentation(inputBlockSize, inputTreshold):
    print "segmentation"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return Segmentation.principal(image, inputBlockSize, inputTreshold)
