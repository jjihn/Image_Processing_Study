import numpy as np
import cv2

img = cv2.imread("me.jpg")
rows, cols = img.shape[0:2] #영상 크기

dx, dy = 100, 50 #이동할 픽셀 거리

# 1. 변환 행렬 생성
mtrx = np.float32([[1,0,dx],
                   [0,1,dy]])
# 2. 단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))

# 3. 탈락된 외각 픽셀을 파랑으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0))

# 4. 탈락된 외각 픽셀을 원본 반사시켜 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, \
                      cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow("Original", img)
cv2.imshow("trans",dst)
cv2.imshow("Border_constant", dst2)
cv2.imshow("Border_reflect", dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()
