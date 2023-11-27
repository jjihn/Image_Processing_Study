import numpy as np
import cv2, time

img = cv2.imread("me.jpg")

#케니 엣지 적용
edge = cv2.Canny(img,100,200)

cv2.imshow("Original",img)
cv2.imshow("Canny", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
