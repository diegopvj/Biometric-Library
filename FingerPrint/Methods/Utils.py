# encoding: utf-8
from PIL import Image, ImageStat
from math import sqrt
import json

sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

def gradient_orientation(image, blockSize):
    size = get_size(image)
    coordinate = get_coordinate(size)
    getPixel = get_pixel(image, (coordinate['x'], coordinate['y']))
    sobelCoordinate = get_sobel_coordinates(sobelOperator)

    result = [[] for i in range(1, coordinate['x'], blockSize)]

    for i in range(1, coordinate['x'], blockSize):
        for j in range(1, coordinate['y'], blockSize):
            print('i,j', (i,j))
            # print('i', i)
            # print('j', j)
    return result

def convert_to_black_and_white(image):
    return image.convert('L')

def get_coordinate(size):
    return json.loads(size)

def get_pixel(image, (coordinateX, coordinateY)):
    imageLoaded = image_load(image)
    return lambda coordinateX, coordinateY: imageLoaded[coordinateX, coordinateY]

def get_size(image):
    (x,y) = image.size
    size = json.dumps({'x': x, 'y': y})
    return size

def get_sobel_coordinates(sobelOperator):
    sobel = json.dumps({'xSobel': transpose(sobelOperator), 'ySobel': sobelOperator})
    return json.loads(sobel)

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

def show_image(image):
    return image.show()

def transpose(matrix):
    transposedMatrix = list(zip(*matrix))
    return transposedMatrix