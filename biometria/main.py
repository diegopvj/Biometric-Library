# encoding: utf-8

import methods.orientation_field
import methods.normalization

def orientation_field(blockSize):
    image = "D:\\REPOSITORIO PROJETO FINAL\\REPO\\biometria\\images\\fingerprint.jpg"
    print "main"
    return methods.orientation_field.principal(image, blockSize)