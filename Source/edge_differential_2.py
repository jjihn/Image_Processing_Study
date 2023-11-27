import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("me.jpg")

# 미분 커널 생성
gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1], [1]])

# 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

# 이미지를 서브플롯으로 표시
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

# 엣지 감지 결과 (GX)
plt.subplot(1, 3, 2)
plt.title("Edge Detection (GX)")
plt.imshow(cv2.cvtColor(edge_gx, cv2.COLOR_BGR2RGB))
plt.axis("off")

# 엣지 감지 결과 (GY)
plt.subplot(1, 3, 3)
plt.title("Edge Detection (GY)")
plt.imshow(cv2.cvtColor(edge_gy, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
