import argparse

def passingArg(image):

    parser = argparse.ArgumentParser(description="Campo de Orientação")

    parser.add_argument(image, nargs=1, help = "Path da imagem")

    args = parser.parse_args()
    return args