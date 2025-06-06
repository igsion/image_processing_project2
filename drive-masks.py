import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
import numpy as np

# image = cv2.imread('./data/DRIVE/training/mask/21_training_mask.gif', cv2.IMREAD_GRAYSCALE)
gif = Image.open('./data/DRIVE/training/mask/21_training_mask.gif')
frames = [np.array(frame.convert('L')) for frame in ImageSequence.Iterator(gif)]
print(frames)

image = np.array(frames[1])
for i in range(len(image)):
    for j in range(len(image[0])):
        if image[i][j] != 0 and image[i][j] != 255:
            print(image[i][j])

# image = cv2.Canny(image, 1, 5)
# plt.imshow(image, cmap='gray')
# plt.show()