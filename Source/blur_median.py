import numpy as np
import cv2

image = cv2.imread("me.jpg")

#미디안 블러 적용
blur = cv2.medianBlur(image, 5)

cv2.imshow("Image", image)
cv2.imshow("Blur", blur)
cv2.waitKey(0)
