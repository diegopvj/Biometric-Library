# encoding: utf-8
from PIL import Image, ImageDraw, ImageStat
import math
import json
import itertools

def calculate_denominator(Gx, Gy):
    return Gx ** 2 - Gy ** 2

def calculate_nominator(Gx, Gy):
    return 2 * Gx * Gy

def calculate_orientation_in_each_block(blockSize, imageSize, sobel, setPixel):
    blockOrientation = [[] for blockXIndex in range(1, imageSize['x'], blockSize)]
    # print(blockOrientation)
    for blockXIndex, blockYIndex in itertools.product(range(1, imageSize['x'], blockSize), range(1, imageSize['y'], blockSize)):
        gradientMagnitudeNominator = 0
        gradientMagnitudeDenominator = 0
        
        for pixelInBlockRow, pixelInBlockColumn in itertools.product(range(blockXIndex, min(blockXIndex + blockSize , imageSize['x'] - 1)), range(blockYIndex, min(blockYIndex + blockSize, imageSize['y'] - 1))):
            Gx = set_mask(setPixel, sobel['x'], pixelInBlockRow, pixelInBlockColumn)
            Gy = set_mask(setPixel, sobel['y'], pixelInBlockRow, pixelInBlockColumn)
            gradientMagnitudeNominator += calculate_nominator(Gx, Gy)
            gradientMagnitudeDenominator += calculate_denominator(Gx, Gy)
        
        gradientMagnitudeInEachPixel = (math.pi + math.atan2(gradientMagnitudeNominator, gradientMagnitudeDenominator)) / 2
        blockOrientation[(blockXIndex - 1) / blockSize].append(gradientMagnitudeInEachPixel)
        # print(blockOrientation)
    return blockOrientation

def convert_to_black_and_white(image):
    return image.convert('L')

def convert_to_RGB(image):
    return image.convert("RGB")

def make_lines(image, imageSize, orientationInEachBlock, blockSize):
    newImage = convert_to_RGB(image)
    imageDraw = ImageDraw.Draw(newImage)

    for blockXIndex, blockYIndex in itertools.product(range(1, imageSize['x'], blockSize), range(1, imageSize['y'], blockSize)):
        orientationAngle = orientationInEachBlock[(blockXIndex - 1) / blockSize][(blockYIndex - 1) / blockSize]
        tangent = math.tan(orientationAngle)
        (x0y0, xy) = get_coordinates_from_line_limits(blockXIndex, blockYIndex, blockSize, tangent)
        
        imageDraw.line([x0y0, xy], fill=250)

    del imageDraw

    return newImage

def get_coordinates_from_line_limits(blockXIndex, blockYIndex, blockSize, tangent):
    if -1 <= tangent and tangent <= 1:
        x0y0 = (blockXIndex, (-blockSize/2) * tangent + blockYIndex + blockSize/2)
        xy = (blockXIndex + blockSize, (blockSize/2) * tangent + blockYIndex + blockSize/2)
    else:
        x0y0 = (blockXIndex + blockSize/2 + blockSize/(2 * tangent), blockYIndex + blockSize/2)
        xy = (blockXIndex + blockSize/2 - blockSize/(2 * tangent), blockYIndex - blockSize/2)
    return (x0y0, xy)

def get_size(image):
    (x,y) = image.size
    size = json.dumps({'x': x, 'y': y})
    return json.loads(size)

def image_load(image):
    return image.load()

def image_mean(image):
    statistics = ImageStat.Stat(image)
    mean = statistics.mean[0]
    return mean

def image_standard_deviation(image):
    statistics = ImageStat.Stat(image)
    standardDeviation = statistics.stddev[0]
    return standardDeviation

def normalize_pixel(x, inputVariance, variance, inputMean, mean):
    dev_coeff = sqrt((inputVariance * ((x - mean)**2)) / variance)
    if x > mean:
        return inputMean + dev_coeff
    return inputMean - dev_coeff

def open_image(image):
    imageOpened = Image.open(image)
    return imageOpened

def save_image(image, imgOut):
	imageSaved = image.save('images/' + imgOut + '.jpg')
	return imageSaved    

def set_mask(setPixel, sobel, pixelInBlockRow, pixelInBlockColumn):
    sobelDimension = len(sobel)
    pixelFiltred = 0
    
    for pixelInSobelRow, pixelInSobelColumn in itertools.product(range(0, sobelDimension), range(0, sobelDimension)):
        pixel = setPixel(pixelInBlockRow + pixelInSobelRow - sobelDimension / 2, pixelInBlockColumn + pixelInSobelColumn - sobelDimension / 2)
        pixelFiltred += pixel * sobel[pixelInSobelRow][pixelInSobelColumn]
    
    return pixelFiltred

def set_pixel(image, (coordinateX, coordinateY)):
    imageLoaded = image_load(image)
    return lambda coordinateX, coordinateY: imageLoaded[coordinateX, coordinateY]

def show_image(image):
    return image.show()

def transpose(matrix):
    transposedMatrix = list(zip(*matrix))
    return transposedMatrix