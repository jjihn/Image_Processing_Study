import numpy as np
import cv2

img = cv2.imread("me.jpg")

# 로버츠 커널 생성 -1
gx_kernel = np.array([[1,0], [0,-1]])
gy_kernel = np.array([[0,1], [-1,0]])

# 커널 적용 -2
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

merged = np.hstack((img, edge_gx, edge_gy, edge_gx + edge_gy))
cv2.imshow("roberts cross", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
