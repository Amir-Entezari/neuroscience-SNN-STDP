from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

class CustomImage:
    def __init__(self, image_path):
        self.image = Image.open(image_path)

    def img2tensor(self):
        # Convert the image to grayscale if it's not already in grayscale
        if self.image.mode != 'L':
            self.image = self.image.convert('L')

        # Convert the image to a NumPy array
        image_array = np.array(self.image)

        # Convert the NumPy array to a Torch tensor
        image_tensor = torch.from_numpy(image_array)

        # Flatten the tensor into a vector
        image_vector = image_tensor.view(-1)

        return image_vector

    def resize_image(self, new_width, new_height):
        # Resize the image
        self.image = self.image.resize((new_width, new_height))

        return self.image

    def img_show(self):
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.show()

# # Example usage:
# image_path = '../dataset/slope.tif'
# vector = img2tensor(image_path)
# print("Shape of the image vector:", type(vector))
