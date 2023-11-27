import numpy as np
import cv2

img = cv2.imread("me.jpg")
height, width = img.shape[:2]

# 1. 크기 지정으로 축소
dst1 = cv2.resize(img, (int(width*0.5), int(height*0.5)), \
                  interpolation=cv2.INTER_AREA)

# 2. 배율 지정으로 확대
dst2 = cv2.resize(img, None, None, 2, 2, cv2.INTER_CUBIC)

cv2.imshow("Original", img)
cv2.imshow("Small", dst1)
cv2.imshow("Big", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
