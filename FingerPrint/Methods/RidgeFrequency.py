import Utils
import OrientationField

def principal(image, inputBlockSize):
    print "depois"
    originalImage = Utils.open_image(image)
    blockSize = int(inputBlockSize)
    imageConverted = Utils.convert_to_black_and_white(originalImage)
    showImage = Utils.show_image(imageConverted)
    gradientOrientation = orientation_field.gradient_orientation_extraction_impl(imageConverted, blockSize)