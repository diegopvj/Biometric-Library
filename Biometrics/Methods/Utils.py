# encoding: utf-8

from PIL import Image, ImageDraw, ImageStat
import math
import json
import itertools


def block_frequency(i, j, blockSize, orientationInEachBlock, im_load):
    tangent = math.tan(orientationInEachBlock)
    orthoTangent = -1 / tangent

    (xNormal, yNormal, step) = steps_through_vector(tangent, blockSize)
    (xCornerCoordinate, yCornerCoordinate) = (0 if xNormal >= 0 else blockSize, 0 if yNormal >= 0 else blockSize)

    greyLevels = []

    for pixelInBlock in range(0, blockSize):
        line = lambda x: (x - xNormal * pixelInBlock * step - xCornerCoordinate) * orthoTangent + yNormal * pixelInBlock * step + yCornerCoordinate
        points = line_points(line, blockSize)
        grayLevel = 0
        for point in points:
            grayLevel += im_load[point[0] + i * blockSize, point[1] + j * blockSize]
        greyLevels.append(grayLevel)

    divisions = len(greyLevels)
    count = detect_peaks_through_grey_levels(greyLevels)

    return count / divisions if divisions > 0 else 0

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

def detect_peaks_through_grey_levels(greyLevels):
    divisions = len(greyLevels)
    treshold = 100
    peak = False
    lastLevel = 0
    lastBottomLevel = 0
    count = 0.0
    for grayLevel in greyLevels:
        if grayLevel < lastBottomLevel:
            lastBottomLevel = grayLevel
        if peak and grayLevel < lastLevel:
            peak = False
            if lastBottomLevel + treshold < lastLevel:
                count += 1
                lastBottomLevel = lastLevel
        if grayLevel > lastLevel:
            peak = True
        lastLevel = grayLevel
    
    return count

def image_frequencys(imageSize, imageLoad, blockSize, orientationInEachBlock):
    frequencys = [[0] for blockRow in range(0, imageSize['x'] / blockSize)]
    for blockRow in range(1, imageSize['x'] / blockSize - 1):
        for blockColumn in range(1, imageSize['y'] / blockSize - 1):
            frequency = block_frequency(blockRow, blockColumn, blockSize, orientationInEachBlock[blockRow][blockColumn], imageLoad)
            frequencys[blockRow].append(frequency)
        frequencys[blockRow].append(0)
    return frequencys

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

def image_copy(image):
    return image.copy()

def image_load(image):
    return image.load()

def image_mean(image):
    statistics = ImageStat.Stat(image)
    mean = statistics.mean[0]
    return mean

def image_new(mode, (width, height), color):
    return Image.new(mode,(width, height), color)

def image_standard_deviation(image):
    statistics = ImageStat.Stat(image)
    standardDeviation = statistics.stddev[0]
    return standardDeviation

def image_draw(image):
    return ImageDraw.Draw(image)

def line_points(line, blockSize):
    newBlock = Image.new("L", (blockSize, 3 * blockSize), 100)
    imageDraw = image_draw(newBlock)
    imageDraw.line([(0, line(0) + blockSize), (blockSize, line(blockSize) + blockSize)], fill=10)
    imageLoad = image_load(newBlock)

    points = []
    for pixelInBlockRow, pixelInBlockColumn in itertools.product(range(0, blockSize), range(0, 3 * blockSize)):
        if imageLoad[pixelInBlockRow, pixelInBlockColumn] == 10:
            points.append((pixelInBlockRow, pixelInBlockColumn - blockSize))
    
    del imageDraw
    del newBlock

    dist = lambda (x, y): (x - blockSize / 2) ** 2 + (y - blockSize / 2) ** 2

    return sorted(points, cmp = lambda x, y: dist(x) < dist(y))[:blockSize]

def make_lines(image, imageSize, orientationInEachBlock, blockSize):
    # imageSize = get_size(image)
    newImage = convert_to_RGB(image)
    imageDraw = image_draw(newImage)

    for blockXIndex, blockYIndex in itertools.product(range(1, imageSize['x'], blockSize), range(1, imageSize['y'], blockSize)):
        orientationAngle = orientationInEachBlock[(blockXIndex - 1) / blockSize][(blockYIndex - 1) / blockSize]
        tangent = math.tan(orientationAngle)
        (x0y0, xy) = get_coordinates_from_line_limits(blockXIndex, blockYIndex, blockSize, tangent)
        
        imageDraw.line([x0y0, xy], fill=250)

    del imageDraw

    return newImage

def normalize_pixel(x, inputVariance, variance, inputMean, mean):
    deviationCoeff = math.sqrt((inputVariance * ((x - mean)**2)) / variance)
    if x > mean:
        return inputMean + deviationCoeff
    return inputMean - deviationCoeff

def open_image(image):
    imageOpened = Image.open(image)
    return imageOpened

def ridge_frequency(image, blockSize, orientationInEachBlock):
    imageSize = get_size(image)
    imageLoad = image_load(image)
    frequencys = image_frequencys(imageSize, imageLoad, blockSize, orientationInEachBlock)
    ridgeFrequencyExtracted = image.copy()

    for blockRow, blockColumn in itertools.product(range(1, imageSize['x'] / blockSize - 1), range(1, imageSize['y'] / blockSize - 1)):
        left = blockRow * blockSize
        top = blockColumn * blockSize
        right = min(blockRow * blockSize + blockSize, imageSize['x'])
        bottom = min(blockColumn * blockSize + blockSize, imageSize['y'])
        box = (left, top, right, bottom)
        ridgeFrequencyExtracted.paste(frequencys[blockRow][blockColumn] * 255.0 * 1.2, box)

    return ridgeFrequencyExtracted 

def segmentation(image, imageSize, inputBlockSize, threshold, segmentedImage, varianceImage):
    for blockXIndex, blockYIndex in itertools.product(range(0, imageSize['x'], inputBlockSize), range(0, imageSize['y'], inputBlockSize)):
        left = blockXIndex
        top = blockYIndex
        right = min(blockXIndex + inputBlockSize, imageSize['x'])
        bottom = min(blockYIndex + inputBlockSize, imageSize['y'])
        block = (left, top, right, bottom)
        croppedBlock = image.crop(block)
        blockStandardDeviation = image_standard_deviation(croppedBlock)
        varianceImage.paste(blockStandardDeviation, block)
        if blockStandardDeviation < threshold:
            segmentedImage.paste(0, block)
    return (segmentedImage, varianceImage)


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

def steps_through_vector(tangent, blockSize):
    (x0y0, xy) = get_coordinates_from_line_limits(0, 0, blockSize, tangent)
    (xVec, yVec) = (xy[0] - x0y0[0], xy[1] - x0y0[1])
    length = math.hypot(xVec, yVec)
    (xNormal, yNormal) = (xVec / length, yVec / length)
    step = length / blockSize

    return (xNormal, yNormal, step)

def transpose(matrix):
    transposedMatrix = list(zip(*matrix))
    return transposedMatrix