from PIL import Image

img = Image.open("./images/IMG_0176.jpg")

img.thumbnail((1024, 1024))

img.save("./images/resized.jpg", quality=85)