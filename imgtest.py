import cv2
import numpy as np

img1 = cv2.imread('drunk.jpg')
bytes1 = img1.tobytes()
print(type(img1), img1.shape)

# img2 = np.frombuffer(bytes1, dtype=np.uint8).reshape([4000, 3000, 3])[:, :, ::-1]
# print(len(bytes1))
# print(type(img2), img2.shape)
# print(img1 == img2)

# import zlib
# compressed_data = zlib.compress(bytes1)
# print(type(compressed_data),len(compressed_data))
# print(len(bytes1))

img_str = cv2.imencode('.jpg', img1)[1].tobytes()
print(type(img_str), len(img_str), len(bytes1))
image = np.asarray(bytearray(img_str), dtype="uint8")
img3 = cv2.imdecode(image, cv2.IMREAD_COLOR)
print(img3 == img1)
print(type(img3), img3.shape)
# cv2.imshow('img_decode',image)
# cv2.waitKey()
