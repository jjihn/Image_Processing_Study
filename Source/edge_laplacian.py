import numpy as np
import cv2

img = cv2.imread("me.jpg")

#라플라시안 필터 적용
edge = cv2.Laplacian(img,-1)

merged = np.hstack((img,edge))
cv2.imshow("Laplacian", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
