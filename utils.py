from PIL import Image

def convert_to_black_and_white(image):
    return image.convert("L")

def open_image(image):
    im = Image.open(image)
    return im

def show_image(image):
    print(image)
    return image.show()
	
def save_image(image, imgOut):
	im = image.save("images/" + imgOut + ".jpg")
	return im