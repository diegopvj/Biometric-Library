# encoding: utf-8

from PIL import Image, ImageDraw, ImageStat
import math
import json
import itertools
import copy

thinning = False


def apply_function_on_each_pixel(orientations, func):
    for x in range(0, len(orientations)):
        for y in range(0, len(orientations[x])):
            orientations[x][y] = func(orientations[x][y])

def apply_structuring_elements(pixels, elements):
    thinning = False
    for element in elements:
        thinning |= merge_element(pixels, element, flatten(element).count(1))

    return thinning;

def apply_values_to_pixels(image, imageSize, pixels):
    imageLoaded = image_load(image)

    for pixelInImageRow, pixelInImageColumn in itertools.product(range(0, imageSize['x']), 
    range(0, imageSize['y'])):
        imageLoaded[pixelInImageRow, pixelInImageColumn] = pixels[pixelInImageRow][pixelInImageColumn]

def block_frequency(blockRowIndex, blockColumnIndex, blockSize, orientationInEachBlock, imageLoaded):
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
            grayLevel += imageLoaded[point[0] + blockRowIndex * blockSize, point[1] + blockColumnIndex * blockSize]
        greyLevels.append(grayLevel)

    divisions = len(greyLevels)
    count = detect_peaks_through_grey_levels(greyLevels)

    return count / divisions if divisions > 0 else 0;

def calculate_denominator(Gx, Gy):
    return Gx ** 2 - Gy ** 2;

def calculate_nominator(Gx, Gy):
    return 2 * Gx * Gy;

def calculate_orientation_in_each_block(blockSize, imageSize, sobel, setPixel):
    blockOrientation = [[] for blockXIndex in range(1, imageSize['x'], blockSize)]
    
    for blockXIndex, blockYIndex in itertools.product(range(1, imageSize['x'], blockSize), 
    range(1, imageSize['y'], blockSize)):
        gradientMagnitudeNominator = 0
        gradientMagnitudeDenominator = 0
        
        for pixelInBlockRow, pixelInBlockColumn in itertools.product(range(blockXIndex, min(blockXIndex + blockSize , imageSize['x'] - 1)), 
        range(blockYIndex, min(blockYIndex + blockSize, imageSize['y'] - 1))):
            Gx = set_mask(setPixel, sobel['x'], pixelInBlockRow, pixelInBlockColumn)
            Gy = set_mask(setPixel, sobel['y'], pixelInBlockRow, pixelInBlockColumn)
            gradientMagnitudeNominator += calculate_nominator(Gx, Gy)
            gradientMagnitudeDenominator += calculate_denominator(Gx, Gy)
        
        gradientMagnitudeInEachPixel = (math.pi + math.atan2(gradientMagnitudeNominator, gradientMagnitudeDenominator)) / 2
        blockOrientation[(blockXIndex - 1) / blockSize].append(gradientMagnitudeInEachPixel)
    
    return blockOrientation;

def convert_to_black_and_white(image):
    return image.convert('L');

def convert_to_RGB(image):
    return image.convert("RGB");

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
    
    return count;

def flatten(matrix):
    return reduce(lambda x, y: x + y, matrix, []);

def gabor_filter(imageConverted, imageLoaded, imageSize, blockSize, xSigma, ySigma, orientationInEachBlock, frequencys):
    gaussKernel = get_gauss_kernel(3)
    merge_kernel(frequencys, gaussKernel)

    for blockRowIndex, blockColumnIndex in itertools.product(range(1, imageSize['x'] / blockSize - 1), 
    range(1, imageSize['y'] / blockSize - 1)):     
        kernel = gabor_kernel(blockSize, orientationInEachBlock[blockRowIndex][blockColumnIndex],
        frequencys[blockRowIndex][blockColumnIndex], xSigma, ySigma)
        
        for pixelInBlockRow, pixelInBlockColumn in itertools.product(range(0, blockSize), 
        range(0, blockSize)): 
            imageLoaded[blockRowIndex * blockSize + pixelInBlockRow,
            blockColumnIndex * blockSize + pixelInBlockColumn] = set_mask(
                lambda x, y: imageLoaded[x, y],
                kernel,
                blockRowIndex * blockSize + pixelInBlockRow,
                blockColumnIndex * blockSize + pixelInBlockColumn)
    
    return imageConverted;

def gabor_kernel(blockSize, orientation, frequency, xSigma, ySigma):
    cos = math.cos(orientation)
    sin = math.sin(orientation)

    xAngle = lambda x, y: -x * sin + y * cos
    yAngle = lambda x, y: x * cos + y * sin

    return kernel_from_func(blockSize, lambda x, y:
        math.exp(-(
            (xAngle(x, y) ** 2) / (xSigma ** 2) +
            (yAngle(x, y) ** 2) / (ySigma ** 2)) / 2) *
        math.cos(2 * math.pi * frequency * xAngle(x, y)));

def gauss_func(x, y):
    sigma = 1
    
    return (1 / (2 * math.pi * sigma)) * math.exp(-(x * x + y * y) / (2 * sigma));

def get_coordinates_from_line_limits(blockXIndex, blockYIndex, blockSize, tangent):
    if -1 <= tangent and tangent <= 1:
        x0y0 = (blockXIndex, (-blockSize/2) * tangent + blockYIndex + blockSize/2)
        xy = (blockXIndex + blockSize, (blockSize/2) * tangent + blockYIndex + blockSize/2)
    else:
        x0y0 = (blockXIndex + blockSize/2 + blockSize/(2 * tangent), blockYIndex + blockSize/2)
        xy = (blockXIndex + blockSize/2 - blockSize/(2 * tangent), blockYIndex - blockSize/2)
    
    return (x0y0, xy);

def get_gauss_kernel(kernelSize):
    return kernel_from_func(kernelSize, gauss_func);

def get_pixels_in_image_loaded(image, imageSize):
    imageLoaded = image_load(image)

    result = []
    for pixelInImageRow in range(0, imageSize['x']):
        result.append([])
        for pixelInImageColumn in range(0, imageSize['y']):
            result[pixelInImageRow].append(imageLoaded[pixelInImageRow, pixelInImageColumn])

    return result;

def get_size(image):
    (x,y) = image.size
    size = json.dumps({'x': x, 'y': y})
    
    return json.loads(size);

def image_copy(image):
    return image.copy();

def image_draw(image):
    return ImageDraw.Draw(image);

def image_frequencys(imageSize, imageLoaded, blockSize, orientationInEachBlock):
    frequencys = [[0] for blockRowIndex in range(0, imageSize['x'] / blockSize)]
    
    for blockRowIndex in range(1, imageSize['x'] / blockSize - 1):
        for blockColumnIndex in range(1, imageSize['y'] / blockSize - 1):
            frequency = block_frequency(blockRowIndex, blockColumnIndex, blockSize, orientationInEachBlock[blockRowIndex][blockColumnIndex], imageLoaded)
            frequencys[blockRowIndex].append(frequency)
        frequencys[blockRowIndex].append(0)
    
    frequencys[0] = frequencys[-1] = [0 for blockRowIndex in range(0, imageSize['y'] / blockSize)]

    return frequencys;

def image_load(image):
    return image.load();

def image_mean(image):
    statistics = ImageStat.Stat(image)
    mean = statistics.mean[0]
    
    return mean;

def image_new(mode, (width, height), color):
    return Image.new(mode,(width, height), color);

def image_standard_deviation(image):
    statistics = ImageStat.Stat(image)
    standardDeviation = statistics.stddev[0]
    
    return standardDeviation;

def kernel_from_func(kernelSize, func):
    kernel = [[] for i in range(0, kernelSize)]
    
    for x, y in itertools.product(range(0, kernelSize), 
    range(0, kernelSize)):
        kernel[x].append(func(x - kernelSize / 2, y - kernelSize / 2))
    
    return kernel;

def line_points(line, blockSize):
    newBlock = Image.new("L", (blockSize, 3 * blockSize), 100)
    imageDraw = image_draw(newBlock)
    imageDraw.line([(0, line(0) + blockSize), (blockSize, line(blockSize) + blockSize)], fill=10)
    imageLoaded = image_load(newBlock)

    points = []
    
    for pixelInBlockRow, pixelInBlockColumn in itertools.product(range(0, blockSize), 
    range(0, 3 * blockSize)):
        if imageLoaded[pixelInBlockRow, pixelInBlockColumn] == 10:
            points.append((pixelInBlockRow, pixelInBlockColumn - blockSize))
    
    del imageDraw
    del newBlock

    distance = lambda (x, y): (x - blockSize / 2) ** 2 + (y - blockSize / 2) ** 2

    return sorted(points, cmp = lambda x, y: distance(x) < distance(y))[:blockSize];

def make_lines(image, imageSize, orientationInEachBlock, blockSize):
    newImage = convert_to_RGB(image)
    imageDraw = image_draw(newImage)

    for blockXIndex, blockYIndex in itertools.product(range(1, imageSize['x'], blockSize), 
    range(1, imageSize['y'], blockSize)):
        orientationAngle = orientationInEachBlock[(blockXIndex - 1) / blockSize][(blockYIndex - 1) / blockSize]
        tangent = math.tan(orientationAngle)
        (x0y0, xy) = get_coordinates_from_line_limits(blockXIndex, blockYIndex, blockSize, tangent)
        
        imageDraw.line([x0y0, xy], fill=250)

    del imageDraw

    return newImage;

def merge_element(pixels, element, result):
    global thinning
    thinning = False

    def pick(old, new):
        global thinning
        if new == result:
            thinning = True
            return 0.0
        return old

    merge_kernel_with_func(pixels, element, pick)

    return thinning;

def merge_kernel(pixel, kernel):
    merge_kernel_with_func(pixel, kernel, lambda old, new: new)    

def merge_kernel_with_func(pixel, kernel, func):
    size = len(kernel)
    
    for x in range(size / 2, len(pixel) - size / 2):
        for y in range(size / 2, len(pixel[x]) - size / 2):
            pixel[x][y] = func(pixel[x][y], set_mask(lambda x, y: pixel[x][y], kernel, x, y))

def normalize_pixel(x, v0, variance, m0, mean):
    deviationCoeff = math.sqrt((v0 * ((x - mean)**2)) / variance)
    
    if x > mean:
        return m0 + deviationCoeff;
    
    return m0 - deviationCoeff;

def open_image(image):
    imageOpened = Image.open(image)
    
    return imageOpened;

def reverse(element):
    copyElement = element[:]
    copyElement.reverse()

    return copyElement;

def ridge_frequency(image, blockSize, orientationInEachBlock):
    imageSize = get_size(image)
    imageLoaded = image_load(image)
    frequencys = image_frequencys(imageSize, imageLoaded, blockSize, orientationInEachBlock)
    ridgeFrequencyExtracted = image.copy()

    for blockRowIndex, blockColumnIndex in itertools.product(range(1, imageSize['x'] / blockSize - 1), 
    range(1, imageSize['y'] / blockSize - 1)):
        left = blockRowIndex * blockSize
        top = blockColumnIndex * blockSize
        right = min(blockRowIndex * blockSize + blockSize, imageSize['x'])
        bottom = min(blockColumnIndex * blockSize + blockSize, imageSize['y'])
        box = (left, top, right, bottom)
        ridgeFrequencyExtracted.paste(frequencys[blockRowIndex][blockColumnIndex] * 255.0 * 1.2, box)

    return ridgeFrequencyExtracted;

def segmentation(image, imageSize, inputBlockSize, threshold, segmentedImage, varianceImage):
    for blockXIndex, blockYIndex in itertools.product(range(0, imageSize['x'], inputBlockSize), 
    range(0, imageSize['y'], inputBlockSize)):
        left = blockXIndex
        top = blockYIndex
        right = min(blockXIndex + inputBlockSize, imageSize['x'])
        bottom = min(blockYIndex + inputBlockSize, imageSize['y'])
        block = (left, top, right, bottom)
        croppedBlock = image.crop(block)
        blockStandardDeviation = image_standard_deviation(croppedBlock)
        varianceImage.paste(blockStandardDeviation ** 2, block)
        if blockStandardDeviation < threshold:
            segmentedImage.paste(0, block)
    
    return (segmentedImage, varianceImage);


def set_mask(setPixel, mask, pixelInBlockRow, pixelInBlockColumn):
    maskDimension = len(mask)
    pixelFiltred = 0
    
    for pixelInMaskRow, pixelInMaskColumn in itertools.product(range(0, maskDimension), 
    range(0, maskDimension)):
        pixel = setPixel(pixelInBlockRow + pixelInMaskRow - maskDimension / 2, pixelInBlockColumn + pixelInMaskColumn - maskDimension / 2)
        pixelFiltred += pixel * mask[pixelInMaskRow][pixelInMaskColumn]
    
    return pixelFiltred;

def set_pixel(image, (coordinateX, coordinateY)):
    imageLoaded = image_load(image)
    
    return lambda coordinateX, coordinateY: imageLoaded[coordinateX, coordinateY];

def show_image(image):
    return image.show();

def smooth_orientations(orientation):
    orientationCos = copy.deepcopy(orientation)
    orientationSin = copy.deepcopy(orientation)
    apply_function_on_each_pixel(orientationCos, lambda x: math.cos(2 * x))
    apply_function_on_each_pixel(orientationSin, lambda x: math.sin(2 * x))

    kernelSize = 5
    gaussKernel = get_gauss_kernel(kernelSize)
    merge_kernel(orientationCos, gaussKernel)
    merge_kernel(orientationSin, gaussKernel)

    for x in range(0, len(orientationCos)):
        for y in range(0, len(orientationCos[x])):
            orientationCos[x][y] = (math.atan2(orientationSin[x][y], orientationCos[x][y])) / 2

    return orientationCos;

def steps_through_vector(tangent, blockSize):
    (x0y0, xy) = get_coordinates_from_line_limits(0, 0, blockSize, tangent)
    (xVec, yVec) = (xy[0] - x0y0[0], xy[1] - x0y0[1])
    length = math.hypot(xVec, yVec)
    (xNormal, yNormal) = (xVec / length, yVec / length)
    step = length / blockSize

    return (xNormal, yNormal, step);

def structuring_elements():
    element1 = [[1, 1, 1], [0, 1, 0], [0.1, 0.1, 0.1]]
    element2 = transpose(element1)
    element3 = reverse(element1)
    element4 = transpose(element3)
    element5 = [[0, 1, 0], [0.1, 1, 1], [0.1, 0.1, 0]]
    element7 = transpose(element5)
    element6 = reverse(element7)
    element8 = reverse(element5)

    thinners = [element1, element2, element3, element4, element5, element6, element7, element8]

    return thinners;

def thinning_structures(imageConverted, imageSize, pixelsInImageLoaded):
    apply_function_on_each_pixel(pixelsInImageLoaded, lambda x: 0.0 if x > 10 else 1.0)
    structuringElements = structuring_elements()
    thinning = True
    
    while(thinning):
        thinning = apply_structuring_elements(pixelsInImageLoaded, structuringElements)
        print "Applying structuring elements"
    
    print "Done Thinning"

    apply_function_on_each_pixel(pixelsInImageLoaded, lambda x: 255.0 * (1 - x))
    apply_values_to_pixels(imageConverted, imageSize, pixelsInImageLoaded)
    
    return imageConverted;

def transpose(matrix):
    transposedMatrix = map(list, zip(*matrix))
    
    return transposedMatrix;