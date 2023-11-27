import numpy as np
import cv2

image = cv2.imread("me.jpg")
if image is None:
    raise Exception("Image load failed!")

# 평균 블러 적용
blurred_image = cv2.blur(image, (5, 5))

cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
