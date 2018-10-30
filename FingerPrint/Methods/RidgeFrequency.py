import Utils
import OrientationField

def principal(image, inputBlockSize):
    print "depois"
    originalImage = Utils.open_image(image)
    blockSize = int(inputBlockSize)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    showImage = Utils.show_image(imageConverted)
    gradientOrientation = OrientationField.gradient_orientation_extraction_impl(imageConverted, blockSize)
    ridgeFrequencyExtraction = ridge_frequency_impl(imageConverted, blockSize, gradientOrientation)
    return ridgeFrequencyExtraction

def ridge_frequency_impl(image, blockSize, angles):
    size = Utils.get_size(image)
    coordinate = Utils.get_coordinate(size)
    frequencys = Utils.frequencys(image, blockSize, angles)
    ridgeFrequencyExtracted = image.copy()