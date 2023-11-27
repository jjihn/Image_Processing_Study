import numpy as np
import cv2

src = cv2.imread("me.jpg")

#aff=np.array([[1,0.5,0],
#              [0, 1, 0]], dtype=np.float32)

aff=np.array([[1,0.1,0],
              [0, 1, 0]], dtype=np.float32)

h,w = src.shape[:2]
#dst = cv2.warpAffine(src, aff, (w+int(h*0.5),h))
dst = cv2.warpAffine(src, aff, (w+int(h*0.1),h))

cv2.imshow("Original", src)
cv2.imshow("Shear transformation", dst)
cv2.waitKey(0)
