import numpy as np
import cv2

img = cv2.imread("me.jpg")
height, width = img.shape[:2]

# 1. 0.5배 축소 변환 행렬
m_small = np.float32([[0.5, 0, 0],
                      [0, 0.5, 0]])

# 2. 2배 확대 변환 행렬
m_big = np.float32([[2, 0, 0],
                    [0, 2, 0]])

# 3. 보간법 적용 없이 확대 축소
dst1 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)))
dst2 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)))

# 4. 보간법 적용한 확대 축소
dst3 = cv2.warpAffine(img, m_small, (int(height*0.5), int(width*0.5)), \
                      None, cv2.INTER_AREA)
dst4 = cv2.warpAffine(img, m_big, (int(height*2), int(width*2)), \
                      None, cv2.INTER_CUBIC)

cv2.imshow("Original", img)
cv2.imshow("Small", dst1)
cv2.imshow("Big", dst2)
cv2.imshow("Small INTER_AREA", dst3)
cv2.imshow("Big INTER_CUBIC", dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()
