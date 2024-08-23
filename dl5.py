###Convolutional Operation
import numpy as np
def conv_operation(img, fil, s):
    img_height, img_width = img.shape
    fil_height, fil_width = fil.shape
    output_image_height = (img_height - fil_height) + 1
    output_image_width = (img_width - fil_width) + 1
    output_image = np.zeros((output_image_height, output_image_width))
    for i in range(0, output_image_height):
        for j in range(0, output_image_width):
###i*s and j*s determine the top left corner of the region within the image.i*s+fil_height and j*s+fil_width determine the bottom right corner of the region.
            region = img[i * s:i * s + fil_height, j * s:j * s + fil_width]
            output_image[i, j] = np.sum(region * fil)
    return output_image
def main():
    # img = np.random.rand(6, 6)
    # fil = np.random.rand(3, 3)
    img = np.array([[3, 0, 1, 2, 7, 4], [1, 5, 8, 9, 3, 1], [2, 7, 2, 5, 1, 3], [0, 1, 3, 1, 7, 8], [4, 2, 1, 6, 2, 8], [2, 4, 5, 2, 3, 9]])
    fil = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    s = 1
    output = conv_operation(img, fil, s)
    print(output)
if __name__=="__main__":
    main()



###Max Pooling
import numpy as np
def max_pool(img, fil, s):
    img_height, img_width = img.shape
    fil_height = fil
    fil_width = fil
    output_image_height = (img_height - fil_height) + 1
    output_image_width = (img_width - fil_width) + 1
    output_image = np.zeros((output_image_height, output_image_width))
    for i in range(0, output_image_height):
        for j in range(0, output_image_width):
            region = img[i*s : i*s+fil_height, j*s : j*s+fil_width]
            output_image[i, j] = np.max(region)
    return output_image
def main():
    # img = np.random.rand(6, 6)
    img = np.array([[1, 3, 2, 1, 3], [2, 9, 1, 1, 5], [1, 3, 2, 3, 2], [8, 3, 5, 1, 0], [5, 6, 1, 2, 9]])
    img = np.reshape(img, (5, 5))
    fil = 3
    s = 1
    output = max_pool(img, fil, s)
    print(output)
if __name__=="__main__":
    main()


