from PIL import Image

def convertToBlackAndWhite(image):
    return image.convert("L")

def openImage(image):
    im = Image.open(image)
    return im

def showImage(image):
    print(image)
    return image.show()