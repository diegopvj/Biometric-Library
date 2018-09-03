import utils
import options

options = options.passingArg('image')

im = utils.open_image(options.image[0])
blockSize = int(options.block_size[0])
imageConverted = utils.convert_to_black_and_white(im)

showImage = utils.show_image(imageConverted)

gradientOrientation = utils.gradient_orientation(imageConverted, blockSize)

if options.name != '':
    imgOut = options.name
    saveImage = utils.save_image(imageConverted, imgOut)