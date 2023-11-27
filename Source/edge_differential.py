import numpy as np
import cv2

img = cv2.imread("me.jpg")

#미분 커널 생성 - 1
gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1],[1]])

#필터 적용 - 2
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)
#결과 출력
merged = np.hstack((img, edge_gx, edge_gy))
cv2.imshow('edge', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
