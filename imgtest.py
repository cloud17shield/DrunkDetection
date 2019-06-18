import cv2
import numpy as np

img1 = cv2.imread('drunk.jpg')
bytes1 = img1.tobytes()
print(type(img1), img1.shape)

img2 = np.frombuffer(bytes1, dtype=np.uint8).reshape(4000, 3000, 3)
print(type(img2), img2.shape)
print(img1 == img2)

