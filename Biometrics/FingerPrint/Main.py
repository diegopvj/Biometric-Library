# encoding: utf-8

from Methods import OrientationField, SmoothOrientationField, Normalization, RidgeFrequency, Segmentation, GaborFilter, Thinning

def orientation_field_extraction(inputBlockSize):
    print "orientation_field"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return OrientationField.main(image, inputBlockSize)

def smooth_orientation_field(inputBlockSize):
    print "smooth_orientation_field"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return SmoothOrientationField.main(image, inputBlockSize)

def normalization(inputMean, inputVariance):
    print "normalization"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    return Normalization.main(image, inputMean, inputVariance)

def ridge_frequency_extraction(inputBlockSize):
    print "ridge_frequency_extraction"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return RidgeFrequency.main(image, inputBlockSize)

def segmentation(inputBlockSize, inputTreshold):
    print "segmentation"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return Segmentation.main(image, inputBlockSize, inputTreshold)

def gabor_filter(inputBlockSize, xSigma, ySigma):
    print "gabor_filter"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return GaborFilter.main(image, inputBlockSize, xSigma, ySigma)

def thinning_structures():
    print "thinning"
    image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\image_8_3\\8_3_gabor.jpg"
    # image = "C:\\Users\\Diego\\Documents\\biometrics-master\\images\\ppf1.png"
    # image = "C:\\Users\\Diego\\Desktop\\digitais\\8_3.tif"
    
    return Thinning.main(image)