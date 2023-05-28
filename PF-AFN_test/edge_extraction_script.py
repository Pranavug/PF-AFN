import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dataset/test_clothes/003434_1.jpg')
OLD_IMG = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)
SIZE = (1, 65)
bgdModle = np.zeros(SIZE, np.float64)

fgdModle = np.zeros(SIZE, np.float64)
rect = (1, 1, img.shape[1], img.shape[0])
cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = mask2[:, :, np.newaxis] * 255

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])

plt.savefig("demo-result.png", bbox_inches='tight', pad_inches=0)

# Function to get edges from clothes image
def get_edges(clothes_image):
    img = clothes_image.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    SIZE = (1, 65)
    bgdModle = np.zeros(SIZE, np.float64)

    fgdModle = np.zeros(SIZE, np.float64)
    rect = (1, 1, img.shape[1], img.shape[0])
    cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = mask2[:, :, np.newaxis] * 255

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
