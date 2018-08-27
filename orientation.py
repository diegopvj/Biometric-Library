import utils
import options

options = options.passingArg("image")

im = utils.openImage(options.image[0])
imageConverted = utils.convertToBlackAndWhite(im)
showImage = utils.showImage(imageConverted)