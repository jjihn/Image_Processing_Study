import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("me.jpg")
rows, cols = img.shape[:2]

# 1. 변환 전, 후 각 3개의 좌표 생성
pts1 = np.float32 ([[100,50], [200,50], [100,200]])
pts2 = np.float32 ([[80,70], [210,60], [250,120]])

# 2. 변환 전 좌표를 이미지에 표시
cv2.circle(img, (100,50), 5, (255,0), -1)
cv2.circle(img, (200,50), 5, (0,255,0), -1)
cv2.circle(img, (100,200), 5, (0,0,255),-1)

# 3. 짝지은 3개의 좌표로 변환행렬 계산
mtrx = cv2.getAffineTransform(pts1, pts2)
# 4. 어핀 변환 적용
dst= cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))

cv2.imshow("Original", img)
cv2.imshow("Affine", dst)
cv2.waitKey(0)
